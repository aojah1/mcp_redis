#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
import re
import json

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient


# ─── LangGraph ReAct agent & supervisor ────────────────
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.language_models.chat_models import BaseChatModel

# ─── Langchains ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from collections import deque

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env
MAIN_FILE    = (PROJECT_ROOT / "mcp_server" / "main.py").resolve()

#────────────────────────────────────────────────────────────────
# 2) Set up LangSmith for LangGraph development
# ────────────────────────────────────────────────────────────────

#from langsmith import Client
#client = Client()
#url = next(client.list_runs(project_name="anup-blog-post")).url
#print(url)
#print("LangSmith Tracing is Enabled")


# ────────────────────────────────────────────────────────
# 3) OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER")
AUTH_TYPE      = "API_KEY"
CONFIG_PROFILE = "DEFAULT"


def initialize_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider=PROVIDER,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 512,
            # remove any unsupported kwargs like citation_types
        },
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

# ────────────────────────────────────────────────────────────────
# 4) Configure Nvidia Nemo Guardrails
# ────────────────────────────────────────────────────────────────
# TBD
def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

#rails_config = RailsConfig.from_content(
#        colang_content=open(get_file_path('nemo_guardrails/rails.config'), 'r').read(),
#        yaml_content=open(get_file_path('nemo_guardrails/config.yml'), 'r').read()
#    )

# ─── NVIDIA Nemo Guardrails spec ──────────────────────────────
# Refuse any politics-related user input
POLITICS_RAIL = """
version: 1
metadata:
  name: no-politics
inputs:
  user_input: str
outputs:
  response: str
completion:
  instructions:
    - when: user_input.lower() matches /(politics|election|government|vote)/
      response: "I’m sorry, I can’t discuss politics."
    - when: true
      response: "{% do %} {{ user_input }} {% enddo %}"
"""
rails_config = RailsConfig.from_content(colang_content=POLITICS_RAIL)
# Initialize the LLM
llm = initialize_llm() # This can be any LLM and need not be the same one used for ReAct
rails = LLMRails(rails_config, llm)

# ────────────────────────────────────────────────────────────────
# 4) Configure MCP Connections to SSE or STDIO
# ────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
# make sure this matches the host+port langraph dev uses (default: 8000)
SSE_HOST = os.getenv("MCP_SSE_HOST", "localhost")
SSE_PORT = os.getenv("MCP_SSE_PORT", "8000")
SERVER_NAME = "redis"
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "sse")


# ────────────────────────────────────────────────────────
## 2. Define your Agent Teams

#Now we can get to define our hierarchical teams. "Choose your player!"

### The team will have ReAct capabilities

#The ReAct team will have a RAG agent and a REDIS "react_agent" as the two worker nodes.
# Let's create those, as well as the team supervisor.
# ────────────────────────────────────────────────────────

class State(MessagesState):
    next: str

def extract_json(text: str) -> dict:
    """
    Extracts a valid JSON object from the text using regex.
    If multiple JSON objects exist, it picks the first one.
    """
    json_pattern = re.findall(r'\{.*?\}', text, re.DOTALL)

    for json_str in json_pattern:
        try:
            parsed_json = json.loads(json_str)

            # Handle nested JSON cases
            if isinstance(parsed_json, dict):
                if "supervisor" in parsed_json and "next" in parsed_json["supervisor"]:
                    return {"next": parsed_json["supervisor"]["next"]}
                if "next" in parsed_json:
                    return parsed_json
        except json.JSONDecodeError:
            continue  # Try the next match if this one fails

    return {"next": "FINISH"}  # Default if no valid JSON is found


######### RAG Agent ###########
# Follow the same template to define your agents in 3 steps

# Step 1. Define the tool
def dummy_tool(input_text):
    return f"Processed: {input_text}"

# Step 2. Define the RAG Agent

def rag_agent():
    tools = [Tool(name="Dummy_RAG_Tool", func=dummy_tool, description="A dummy RAG test tool")]

    rag_agent = initialize_agent(
        tools= tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return rag_agent

# Step 3. Define the RAG Agent Langraph Node
def rag_node(state: State) -> Command[Literal["supervisor"]]:
    # Ensure state is a dictionary
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = state.messages  # If it's an object, use its attribute

    # Extract last user message
    if messages:
        user_input = messages[-1].content
    else:
        raise ValueError("No messages found in state")

    # 🚀 Debugging: Print the extracted message
    print("User Input Received:", user_input)

    # Invoke the agent service
    result = rag_agent().invoke({"input": user_input})

    # 🚀 Debugging: Print the raw agent response
    print("Agent Response:", result)

    # Handle case where result is a string instead of a dictionary
    if isinstance(result, str):
        result = {"output": result}  # Convert to dictionary format

    # Ensure result contains expected keys
    if not isinstance(result, dict):
        raise TypeError(f"Expected a dictionary but got {type(result)} instead.")

    if "output" not in result:
        raise KeyError(f"Expected key 'output' missing. Available keys: {list(result.keys())}")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["output"], name="RAG")
            ]
        },
        goto="supervisor",
    )
########## END RAG AGENT ##################

######### REDIS MCP based Agent ###########
# Step 1. Define the tool
from agents import Agent
def setup_redis():
    server = MCPServerStdio(
        params={
            "command": sys.executable,  # venv’s python 3.13
            "args": [str(MAIN_FILE)],  # plain script, no uvicorn
            "env": {
                "REDIS_HOST": "127.0.0.1",
                "REDIS_PORT": "6379",
                # add auth env vars if you need them
            },
        }
    )
    return server

# Step 3. Define the REDIS Agent
async def redis_agent():
    server = setup_redis()
    await server.connect()

    redis_agent = Agent(
        name="Redis Assistant",
        instructions=("You are a helpful assistant capable of reading and writing to Redis. Store every question and answer in the Redis Stream app:logger."),
        # llm=llm,
        mcp_servers=[server],
    )
    return redis_agent
    #redis_agent = initialize_agent(
    #    llm=llm,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #    mcp_servers=[server],
    #    verbose=True
    #)


# Step 3. Define the REDIS Agent Langraph Node
def redis_node(state: State) -> Command[Literal["supervisor"]]:
    # Ensure state is a dictionary
    if isinstance(state, dict):
        messages = state.get("messages", [])
    else:
        messages = state.messages  # If it's an object, use its attribute

    # Extract last user message
    if messages:
        user_input = messages[-1].content
    else:
        raise ValueError("No messages found in state")

    # 🚀 Debugging: Print the extracted message
    print("User Input Received:", user_input)

    # Invoke the agent service
    result = redis_agent().ainvoke({"input": user_input})

    # 🚀 Debugging: Print the raw agent response
    print("Agent Response:", result)

    # Handle case where result is a string instead of a dictionary
    if isinstance(result, str):
        result = {"output": result}  # Convert to dictionary format

    # Ensure result contains expected keys
    if not isinstance(result, dict):
        raise TypeError(f"Expected a dictionary but got {type(result)} instead.")

    if "output" not in result:
        raise KeyError(f"Expected key 'output' missing. Available keys: {list(result.keys())}")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["output"], name="REDIS")
            ]
        },
        goto="supervisor",
    )

##############################################
# ────────────────────────────────────────────────────────
# 5) build a Supervisor LangGraph agent
# ────────────────────────────────────────────────────────

## Supervisor Agent

def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    """
    Supervisor function responsible for routing user queries to the appropriate LangGraph sub-agent.
    It ensures that:
      - General knowledge-based queries are handled by **RAG Agent**
      - Business related questions are handled by **REDIS**
    """

    options = ["FINISH"] + members
    system_prompt = """
    Supervisor function responsible for routing user queries to the appropriate LangGraph sub-agent.
    It ensures that:
      - General knowledge-based queries are handled by **RAG Agent**
      - Business related questions are handled by **REDIS**

    Your role is to intelligently route queries to the correct sub-agent while ensuring efficiency. 
    Avoid redundant tool calls, and if a tool fails, escalate to the next available option.
    If no relevant tool is found or the conversation is complete, return: {"next": "FINISH"}.

    ### **Available Agents & Responsibilities**
    - **RAG** → Handles **general knowledge questions** that requires public knowledge.
    - **REDIS** → Handles **Business related questions ** that requires database access.
    
    ### **Routing Rules**
    1. **how to create a good recipe** → Route to **RAG**.  
    2. **Business related questions** → Route to **REDIS**.  
    3. **Do not call the same tool twice in succession** unless needed. If a tool fails, escalate to another tool if applicable.  
    4. **If no relevant tool is found, or conversation is complete, return:** {"next": "FINISH"}.

    ### **Examples for Better Routing**
    ❌ **Avoid vague or incorrect routing decisions. Follow these examples:**  

    **Example 1: good recipe**  
    **User:** how to create a good recipe?  
    **Response:** {"next": "RAG"}  

    **Example 2: Scrape a webpage*  
    **User:** Find recent research papers on quantum computing. , [https://arxiv.org/list/quant-ph/recent].  
    **Response:** {"next": "RAG"}  

    **Example 3: Real-Time Query (Internet-based)**  
    **User:** What is the temperature of Louisville KY today?  
    **Response:** {"next": "RAG"}  

    **Example 4: show all formats for invoice numbers based on the record retrieved from  HGETALL 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'**  
    **User:** Get me insights.  
    **Response:** {"next": "REDIS"}  

    **Example 5: Conversation is Complete**  
    **User:** Thanks, that’s all I needed.  
    **Response:** {"next": "FINISH"}  
    """

    class Router(TypedDict):
        """Worker to route to next. If no workers are needed, route to FINISH."""
        next: Literal[tuple(options)]  # Corrected Literal usage

    def supervisor_node(state: State) -> Command[Literal[tuple(members) + ("__end__",)]]:
        """An LLM-based router for LangGraph-based sub-agents."""

        messages = [HumanMessage(content=system_prompt)] + state["messages"]

        # Get LLM response
        response_text = llm.invoke(messages).content  # Ensure we extract .content
        # print(response_text)
        response = extract_json(response_text)  # Use robust JSON extraction
        goto = response.get("next", "FINISH")  # Default to FINISH if missing

        # Ensure valid routing
        if goto not in members:
            print(f"⚠️ route received: {goto}, defaulting to FINISH")
            goto = "FINISH"

        if goto == "FINISH":
            goto = END

        print(f"📌 Routing user to: {goto}")  # Debugging log
        return Command(goto=goto, update={"next": goto})

    return supervisor_node


# Bring all the agents togather - Supervisor Agent
def build_supervisor_agent():
    supervisor_node = make_supervisor_node(llm, ["RAG", "REDIS"])

    ### Now Build the Graph for execution
    # Now that we've created the necessary components, defining their interactions is easy.
    # Add the nodes to the team graph, and define the edges, which determine the transition criteria.

    research_builder = StateGraph(State)
    research_builder.add_node("supervisor", supervisor_node)
    research_builder.add_node("RAG", rag_node)
    research_builder.add_node("REDIS", redis_node)

    research_builder.add_edge(START, "supervisor")

    # Store for long-term (across-thread) memory
    across_thread_memory = InMemoryStore()
    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()

    # Compile the graph with the checkpointer and store
    graph = research_builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)

    return graph

# We can give this team work directly. Try it out below.
#### Memory Management across the multi-agent system

def manage_lt_memory():
    # Initialize in-memory store
    memory_store = InMemoryStore()

    # Define a namespace for memory (can be per user or per conversation)
    user_id = "aojah1"
    memory_namespace = (user_id, "multi_agent_memory")

# Extract content from response dictionary
def print_message(response):
    memory = ""
    if isinstance(response, dict):
        for agent, data in response.items():
            if "messages" in data and data["messages"]:
                # print("Store this in memory::")
                memory = data["messages"][-1].content
    else:
        print("Store this in memory: ")

    return memory


def run_chatbot():
    # Test WITH in -memory - NL2SQL - SQLLITE
    # We supply a thread ID for short-term (within-thread) memory
    # We supply a user ID for long-term (across-thread) memory
    # Enable Chat Capability of the Agent to execute this prompt

    config = {"configurable": {"thread_id": "3", "user_id": "aojah1"}}
    # User input
    graph = build_supervisor_agent()
    # Run the graph
    #question = [HumanMessage(content="where is louisville in USA?")]
    #
    question = [HumanMessage(content="show all formats for invoice numbers based on the record retrieved from  HGETALL 'session:e5f6a932-6123-4a04-98e9-6b829904d27f")]
    for s in graph.stream(
            {"messages": question},
            config,
    ):
        print_message(s)
        print("---")
    #for chunk in graph.stream({"messages": question}, config, stream_mode="values"):
    #    chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    run_chatbot()


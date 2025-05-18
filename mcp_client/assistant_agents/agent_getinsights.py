#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LangGraph ReAct agent & supervisor ────────────────
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from collections import deque
# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

# ─── Utilities ──────────────────────────────
import asyncio, sys, os, logging, re, json
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict
from collections import deque


# ─── init logging & env ─────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

#────────────────────────────────────────────────────────────────
# 2) Set up LangSmith for LangGraph development
# ────────────────────────────────────────────────────────────────

from langsmith import Client
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
llm: BaseChatModel = initialize_llm() # This can be any LLM and need not be the same one used for ReAct
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
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

class State(TypedDict):
    messages: Annotated[list, add_messages]

from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str

# ─── simple JSON extractor for router ───────────────
def extract_json(text: str) -> dict:
    for j in re.findall(r'\{.*?\}', text, re.DOTALL):
        try:
            obj = json.loads(j)
            if "next" in obj or ("supervisor" in obj and "next" in obj["supervisor"]):
                return obj.get("supervisor", obj)
        except:
            pass
    return {"next":"FINISH"}

# ────────────────────────────────────────────────────────
# 5) build all the Agents
# ────────────────────────────────────────────────────────
# ─── REDIS MCP NODE ────────────────────────────────
connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST", "127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
                "TRANSPORT": "stdio",
            },
        }
    }

# Define the logic to call the model
# We'll define a node to call our LLM that incorporates a summary, if it exists, into the prompt.
async def call_model(state: State):
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        inp = [SystemMessage(content=system_message)] + state["messages"]

    else:
        inp = state["messages"]

    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )

        agent = create_react_agent(model=initialize_llm(), tools=tools)
        # invoke with a list of messages, not a dict
        result = await agent.ainvoke({"messages": inp})
        # restore this line so `text` actually exists:
        #text = result.content if isinstance(result, AIMessage) else str(result)

    return {"messages": result["messages"]}

#We'll define a node to produce a summary.
#Note, here we'll use `RemoveMessage` to filter our state after we've produced the summary.
async def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# We'll add a conditional edge to determine whether to produce a summary based on the conversation length.
# Determine whether to end or summarize the conversation
async def should_continue(state: State):
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

# ────────────────────────────────────────────────────────
# 7) BUILD GRAPH & RUNNER
# ────────────────────────────────────────────────────────

async def build_graph():
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import StateGraph, START

    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile()

    return graph


### REPL

async def getinsights(max_history: int = 30):
    graph = await build_graph()
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        output = await graph.ainvoke(
            {"messages": user_text},
            # config
        )
        for m in output["messages"][-1:]:
            m.pretty_print()
# ────────────────────────────────────────────────────────
# 9) Test cases
# ────────────────────────────────────────────────────────

async def run_agent_async():
    graph = await build_graph()
    # Test inputs
    str1 = "delete all record using hdel from redis db"
    str2 = "can you show more details of each invoice number with the format ERS-XXXXX-YYYYYY"

    # Use a flat list of HumanMessage
    config = {"configurable": {"thread_id": "1"}}
    # First query
    input_messages = [HumanMessage(content=str1)]
    output = await graph.ainvoke(
        {"messages": input_messages},
        #config
    )
    for m in output["messages"][-1:]:
        m.pretty_print()

    # Second query
    input_messages = [HumanMessage(content=str2)]
    output = await graph.ainvoke(
        {"messages": input_messages},
        # config
    )
    for m in output["messages"][-1:]:
        m.pretty_print()


if __name__=="__main__":
    asyncio.run(getinsights())

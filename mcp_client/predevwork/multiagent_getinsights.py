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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
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
# ─── RAG NODE ──────────────────────────────────────
def rag_agent():
    tools = [Tool(name="Dummy_RAG_Tool",
                  func=lambda txt: f"Processed: {txt}",
                  description="Dummy RAG tool")]


    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

async def rag_node(state: State) -> Command[Literal["supervisor"]]:
    inp = state["messages"][-1].content
    print("User Input Received (RAG):", inp)

    result = rag_agent().invoke([HumanMessage(content=inp)])
    # .invoke() may return a dict or an AIMessage
    if isinstance(result, dict):
        output = result.get("output", str(result))
    elif hasattr(result, "content"):
        output = result.content
    else:
        output = str(result)

    print("Agent Response (RAG):", output)
    return Command(
        update={"messages":[HumanMessage(content=output,name="RAG")]},
        goto="FINISH"
    )

# ─── REDIS MCP NODE ────────────────────────────────
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
SERVER_NAME = "redis"
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



async def redis_node(state: State) -> Command[Literal["supervisor"]]:
    inp = state["messages"][-1].content

    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )
        print("MCP Tools Found:", tools)
        agent  = create_react_agent(model=initialize_llm(), tools=tools)
        # invoke with a list of messages, not a dict
        result = await agent.ainvoke({"role": "user","messages": inp})
        # restore this line so `text` actually exists:
        text = result.content if isinstance(result, AIMessage) else str(result)
        print("Agent Response (Redis):", text)
    return Command(
        update={"messages": [HumanMessage(content=text, name="REDIS")]},
        goto="FINISH"
    )


# ────────────────────────────────────────────────────────
# 6) build a Supervisor LangGraph agent
# ────────────────────────────────────────────────────────
def make_supervisor_node(members: list[str]):
    system_prompt = """
        Supervisor function responsible for routing user queries to the appropriate LangGraph sub-agent.
        It ensures that:
          - Internet-specific queries are handled by **RAG**
          - Natural language to SQL queries about recipe are handled by **REDIS**

        Your role is to intelligently route queries to the correct sub-agent while ensuring efficiency. 
        Avoid redundant tool calls, and if a tool fails, escalate to the next available option.
        If no relevant tool is found or the conversation is complete, return: {"next": "FINISH"}.

        Route to RAG for general knowledge; to REDIS for HGETALL/invoice/db work.
        Reply with JSON: {"next":"RAG"}, {"next":"REDIS"}, or {"next":"FINISH"}.
        """

    system = SystemMessage(content=system_prompt)

    class Router(TypedDict):
        next: Literal["RAG","REDIS","FINISH"]

    async def supervisor_node(state: State) -> Command:
        prompt = [system] + state["messages"]
        ai: AIMessage = await llm.ainvoke(prompt)
        route = extract_json(ai.content).get("next","FINISH")
        if route not in members:
            route = "FINISH"
        goto = END if route=="FINISH" else route
        print(f"📌 Routing user to: {route}")
        return Command(goto=goto, update={"next":route})

    return supervisor_node

supervisor_node = make_supervisor_node(["RAG", "REDIS"])

# ────────────────────────────────────────────────────────
# 7) BUILD GRAPH & RUNNER
# ────────────────────────────────────────────────────────

async def build_graph():

    graph = StateGraph(State)

    # 1) your real handlers
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("RAG", rag_node)
    graph.add_node("REDIS", redis_node)

    # 2) kick off from the built-in START
    graph.add_edge(START, "supervisor")

    #graph.add_edge("supervisor", "RAG")
    #graph.add_edge("supervisor", "REDIS")

    # 4) finish each branch
    graph.add_edge("supervisor", END)
    graph.add_edge("RAG", END)
    graph.add_edge("REDIS", END)


    # 3) compile without a checkpointer
    return graph.compile(store=InMemoryStore())

# ────────────────────────────────────────────────────────
# 8) xxxxxxxxxx
# ────────────────────────────────────────────────────────
import json
from pprint import pprint
# Extract content from response dictionary
def print_message(response):
    # Only handle dict‐shaped responses
    if not isinstance(response, dict):
        return

    for agent, data in response.items():
        # skip if data is None or not a dict
        if not isinstance(data, dict):
            continue

        msgs = data.get("messages")
        if msgs:
            # print the last message's content
            raw = msgs[-1].content
            try:
                obj = json.loads(raw)
                pprint(obj)  # pretty‐print the parsed JSON
            except json.JSONDecodeError:
                pprint(raw)


            # ────────────────────────────────────────────────────────
# 8) simple REPL
# ────────────────────────────────────────────────────────

async def getinsights(max_history: int = 30):
    graph = await build_graph()
    history: deque[HumanMessage | AIMessage] = deque(maxlen=max_history)

    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        history.append(HumanMessage(content=user_text))
        question = [HumanMessage(content=user_text)]

        async for step in graph.astream(
                {"messages": question},
                {"configurable": {"thread_id": "3", "user_id": "aojah1"}}
        ):
            print_message(step)

            # if there was an AI reply, stash it into history
            if isinstance(step, dict):
                for data in step.values():
                    if isinstance(data, dict) and data.get("messages"):
                        ai_msg = data["messages"][-1]
                        if isinstance(ai_msg, AIMessage):
                            history.append(ai_msg)


# ────────────────────────────────────────────────────────
# 9) Test cases
# ────────────────────────────────────────────────────────

async def run_agent_async():
    graph = await build_graph()
    str1 = "show all formats for invoice numbers where session:e5f6a932-6123-4a04-98e9-6b829904d27f"
    str2 = "where is Lousiville KY?"
    str3 = "show me all the tools from the redis cluster"
    str4 = """HSET session:e5f6a932-6123-4a04-98e9-6b829904d27f record:10 Id "46" Vendor Name "GE Plastics" Invoice Number "ERS-13393-222295" Invoice Type "STANDARD" Amount Due "15,165.74" Past Due Days "98", Id "47" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221806" Invoice Type "STANDARD" Amount Due "22,076.14" Past Due Days "104", Id "48" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221805" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "49" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13373-221916" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "50" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13376-221922" Invoice Type "STANDARD" Amount Due "3,311.42" Past Due Days "105"""""
    question = [HumanMessage(content=(str1))]

    async for step in graph.astream(
        {"role":"user", "messages":question},
        {"configurable":{"thread_id":"3","user_id":"aojah1"}}
    ):
        print_message(step)
        print("---")

if __name__=="__main__":
    asyncio.run(getinsights())

#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from datetime import datetime
from contextlib import asynccontextmanager

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
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from collections import deque

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
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
llm_oci = initialize_llm() # This can be any LLM and need not be the same one used for ReAct
rails = LLMRails(rails_config, llm_oci)

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

connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST", "127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
                "MCP_TRANSPORT": MCP_TRANSPORT,
            },
        }
    }

# ────────────────────────────────────────────────────────
# 5) build a Supervisor LangGraph agent
# ────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]


async def build_agent():
    # configure the single Redis-MCP server
    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )

        # Initialize the LLM
        llm = initialize_llm()
        llm_with_tools = llm.bind_tools(tools)

        SYSTEM_PROMPT = (
            "You are a Redis-savvy assistant. "
            "For reads: always use HGETALL.\n"
            "For writes: use HSET (and EXPIRE when needed)."
        )

        def supervisor(state: State):
            messages = state["messages"]

            # Insert system prompt only once
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

            result = llm_with_tools.invoke(messages)
            return {"messages": [result]}

        # Build LangGraph
        builder = StateGraph(State)
        builder.add_node("supervisor", supervisor)
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("supervisor", tools_condition)
        builder.add_edge("tools", "supervisor")
        builder.add_edge(START, "supervisor")
        builder.add_edge("supervisor", END)

        graph = builder.compile(interrupt_before=[], interrupt_after=[])
        graph.name = "getinsight-supervisor"

        await getinsights(graph)
        return graph


# ────────────────────────────────────────────────────────
# 6) REPL that strips out any non-string AIMessage.content
# ────────────────────────────────────────────────────────
# ─── simple REPL ────────────────────────────────────
async def getinsights(agent, max_history: int = 30):
    history: deque[HumanMessage|AIMessage] = deque(maxlen=max_history)
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        history.append(HumanMessage(content=user_text))
        result = await agent.ainvoke({"messages": list(history)})
        ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        reply = ai_msg.content if ai_msg else "⚠️ (no reply)"
        print(f"\n🤖 {reply}\n")
        history.append(AIMessage(content=reply))

if __name__ == "__main__":
    graph = asyncio.run(build_agent())


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
from langchain_core.messages import HumanMessage, AIMessage

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
# make sure this matches the host+port langraph dev uses (default: 8000)
SSE_HOST = os.getenv("MCP_SSE_HOST", "localhost")
SSE_PORT = os.getenv("MCP_SSE_PORT", "8000")
SERVER_NAME = "redis"
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

if(MCP_TRANSPORT == "stdio"):
    connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST", "127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
            },
        }
    }
else:
    connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args":    [str(MCP_SCRIPT)],
            "transport": MCP_TRANSPORT,
            "url":      f"http://{SSE_HOST}:{SSE_PORT}/sse?server={SERVER_NAME}",
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST","127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT","6379"),
                "MCP_TRANSPORT": MCP_TRANSPORT,
            },
        }
    }

#────────────────────────────────────────────────────────────────
# 2) Set up LangSmith for LangGraph development
# ────────────────────────────────────────────────────────────────

from langsmith import Client
client = Client()
url = next(client.list_runs(project_name="anup-blog-post")).url
print("LangSmith Tracing URL: ")
print(url)

# ────────────────────────────────────────────────────────
# 3) OCI GenAI configuration
# ────────────────────────────────────────────────────────

COMPARTMENT_ID  = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT        = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID        = os.getenv("OCI_GENAI_MODEL_ID")
AUTH_TYPE       = "API_KEY"
CONFIG_PROFILE  = "DEFAULT"

def initialize_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider="cohere",
        model_kwargs={"temperature": 0.5, "max_tokens": 512},
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]

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

# ─── main with MultiServerMCPClient ─────────────────

async def main():
    # configure the single Redis-MCP server
    # start up the MCP server process and connect
    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )

        # build a LangGraph ReAct agent
        llm   = initialize_llm()
        llm_with_tools = llm.bind_tools(tools)

        def supervisor(state: State):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        # build the StateGraph
        builder = StateGraph(State)

        builder.add_node("supervisor", supervisor)
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("supervisor", tools_condition)
        builder.add_edge("tools", "supervisor")
        builder.add_edge(START, "supervisor")
        builder.add_edge("supervisor", END)

        graph = builder.compile(
            interrupt_before=[],  # if you want to update the state before calling the tools
            interrupt_after=[],
        )
        graph.name = "getinsight-supervisor"
        # hand off to your REPL
        await getinsights(graph)
        return graph

if __name__ == "__main__":
    asyncio.run(main())

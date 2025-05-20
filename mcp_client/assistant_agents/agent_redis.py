#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LangGraph ReAct agent & supervisor ────────────────
from langgraph.prebuilt import create_react_agent

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState



# ────────────────────────────────────────────────────────────────
# 1) init logging & env
#────────────────────────────────────────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent

# ────────────────────────────────────────────────────────────────
# 2) Configure MCP Connections to SSE or STDIO
# ────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
# make sure this matches the host+port langraph dev uses (default: 8000)
SSE_HOST = os.getenv("MCP_SSE_HOST", "localhost")
SSE_PORT = os.getenv("MCP_SSE_PORT", "8000")
SERVER_NAME = "redis"
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio")

# ─── REDIS MCP NODE ────────────────────────────────
connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": REDIS_HOST,
                "REDIS_PORT": REDIS_PORT,
                "TRANSPORT": MCP_TRANSPORT,
            },
        }
    }


class State(MessagesState):
    summary: str

async def redis_node(state: State, llm: BaseModel):
    inp = state["messages"][-1].content

    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )

        SYSTEM_PROMPT = (
            "You are a Redis-savvy assistant. "
            "For reads: always use HGETALL.\n"
            "For writes: use HSET (and EXPIRE when needed)."
        )

        messages = state["messages"]

        # Insert system prompt only once
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        agent = create_react_agent(
            model=llm,
            tools=tools,
            name="redis_expert",
            prompt=SYSTEM_PROMPT,
        )
        # invoke with a list of messages, not a dict
        result = await agent.ainvoke({"messages": inp})

    return {"messages": result["messages"]}


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
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LangGraph ReAct agent & supervisor ────────────────
from langgraph.prebuilt import create_react_agent

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import MessagesState

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# ────────────────────────────────────────────────────────────────
# 1) init logging & env
#────────────────────────────────────────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

# ────────────────────────────────────────────────────────────────
# 2) Configure MCP Connections to SSE or STDIO
# ────────────────────────────────────────────────────────────────
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

# ────────────────────────────────────────────────────────────────
# Configure MCP Connections to SSE (Streamable HTTP)
# ────────────────────────────────────────────────────────────────
MCP_TRANSPORT= os.getenv("MCP_TRANSPORT","stdio") #"stdio" #streamable_http" #stdio" #sse
MCP_SSE_HOST=os.getenv("MCP_SSE_HOST","0.0.0.0")
MCP_SSE_PORT=os.getenv("MCP_SSE_PORT","8000")


connections = {
    "redis": {
        "url": f"http://{MCP_SSE_HOST}:{MCP_SSE_PORT}/mcp",
        "transport": MCP_TRANSPORT,
    }
}
print(connections)
# Build your client
client = MultiServerMCPClient(connections)
print(client)
class State(MessagesState):
    summary: str

async def redis_node(state: State, llm: BaseModel):
    inp = state["messages"][-1].content

    # Start a session for the "redis" server
    async with client.session("redis") as session:
        tools = await load_mcp_tools(session)

        SYSTEM_PROMPT = (
            "You are a Redis-savvy assistant. "
            "For reads: always use HGETALL. "
            "For writes: use HSET (and EXPIRE when needed)."
        )

        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

        agent = create_react_agent(
            model=llm,
            tools=tools,
            name="redis_expert",
            prompt=SYSTEM_PROMPT,
        )

        result = await agent.ainvoke({"messages": inp})

    return {"messages": result["messages"]}

# Test Cases -
# now invoke the tool with the “state” envelope:
async def test_case():
    from mcp_client.llm.oci_genai import initialize_llm

    raw_state = {
        "messages": [HumanMessage(content="which Invoice I should pay first based criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'")]
    }

    answer = await redis_node(raw_state, initialize_llm())
    # find the last AIMessage
    ai_reply = next(
        (m for m in reversed(answer["messages"]) if isinstance(m, AIMessage)),
        None
    )

    if ai_reply:
        print("→ AI says:", ai_reply.content)
    else:
        print("→ (no AI reply found)")

if __name__ == "__main__":
    asyncio.run(test_case())
#!/usr/bin/env python3
import asyncio
import sys
import os
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

from agents.mcp import MCPServerStdio
from langchain_community.chat_models import ChatOCIGenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ─────────────── Constants & Paths ───────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MAIN_FILE    = PROJECT_ROOT / "mcp_server" / "main.py"

# ─────────────── Load Env & OCI LLM Setup ───────────────
load_dotenv(PROJECT_ROOT / ".env")

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
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
        model_kwargs={"temperature": 0.5, "max_tokens": 512},
    )

# ─────────────── Build LangGraph Agent ───────────────
async def build_agent():
    # 1) start the Redis-MCP server
    server = MCPServerStdio(
        params={
            "command": sys.executable,
            "args": [str(MAIN_FILE)],
            "env": {
                "REDIS_HOST": "127.0.0.1",
                "REDIS_PORT": "6379",
            },
        }
    )
    # 2) connect to initialize session
    await server.connect()

    # 3) grab the internal session object
    session = getattr(server, "session", None) or getattr(server, "_session", None)
    if session is None:
        raise RuntimeError(f"Could not find MCP session on server; attrs: {dir(server)}")

    # 4) load LangChain tools from the session
    tools = await load_mcp_tools(session)

    # 5) create a ReAct agent seeded with a SystemMessage
    agent = create_react_agent(
        model=initialize_llm(),
        tools=tools,
    )

    return server, agent

# ─────────────── CLI Loop ───────────────
async def cli(agent):
    # initial system prompt
    system = SystemMessage(
        content=(
            "You are a helpful assistant capable of reading and writing to Redis. "
            "Store every question and answer in the Redis Stream app:logger."
        )
    )

    print("🔧 Redis→LangGraph Agent CLI — type 'exit' to quit\n")
    while True:
        q = input("❓> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # build non-empty message list
        msgs = [
            system,
            HumanMessage(content=q),
        ]

        # invoke the agent with the full messages
        result = await agent.ainvoke({"messages": msgs})

        # extract the AIMessage reply
        ai_msg = None
        if isinstance(result, dict) and "messages" in result:
            ai_msg = next((m for m in result["messages"] if isinstance(m, AIMessage)), None)
        answer = ai_msg.content if ai_msg else "⚠️ (no reply)"
        print(f"\n🤖 {answer}\n")

# ─────────────── Main Entry Point ───────────────
async def main():
    server, agent = await build_agent()
    async with server:
        await cli(agent)

if __name__ == "__main__":
    asyncio.run(main())

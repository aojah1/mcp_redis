#!/usr/bin/env python3.13
import os
import sys
import asyncio
import logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# silence Pydantic/langchain_core warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & transports ────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ─── LangGraph ReAct agent ───────────────────────────────
from langgraph.prebuilt import create_react_agent

# ─── OCI LLM ─────────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── LangChain message types ────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage

# ─────────────────────────────────────────────────────────
def initialize_llm():
    """
    Load OCI creds from .env and return a ChatOCIGenAI instance.
    """
    # assumes a PROJECT_ROOT/.env with OCI_GENAI_* and OCI_COMPARTMENT_ID
    load_dotenv(Path(__file__).parent.parent / ".env")
    return ChatOCIGenAI(
        model_id=os.environ["OCI_GENAI_MODEL_ID"],
        service_endpoint=os.environ["OCI_GENAI_ENDPOINT"],
        compartment_id=os.environ["OCI_COMPARTMENT_ID"],
        provider=os.environ.get("PROVIDER", "cohere"),
        model_kwargs={"temperature": 0.5, "max_tokens": 512},
        auth_type="API_KEY",
        auth_profile="DEFAULT",
    )

# ─────────────────────────────────────────────────────────
async def build_agent(session: ClientSession):
    """
    Given an open MCP session, load the Redis tools and return
    a prebuilt ReAct‑style LangGraph agent.
    """
    llm = initialize_llm()
    tools = await load_mcp_tools(session)
    return create_react_agent(
        llm,
        tools,
        prompt=(
            "You are a Redis‑savvy assistant.\n"
            "When asked to read from Redis, always use HGETALL.\n"
            "When asked to write, use HSET or EXPIRE as needed."
        ),
        name="redis-supervisor",
    )

# ─────────────────────────────────────────────────────────
async def getinsights(agent, max_history: int = 30):
    """
    Simple REPL: keeps a short history, sends it to the agent,
    and prints out the AIMessage response.
    """
    print("🔧  Redis Supervisor — type 'exit' to quit\n")
    history: deque[HumanMessage | AIMessage] = deque(maxlen=max_history)

    while True:
        q = input("❓> ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        # add the new user turn
        history.append(HumanMessage(content=q))

        # call the agent
        result = await agent.ainvoke({"messages": list(history)})
        ai_msg = next(
            (m for m in result["messages"] if isinstance(m, AIMessage)),
            None
        )
        reply = ai_msg.content if ai_msg else "⚠️ (no reply)"
        print(f"\n🤖 {reply}\n")

        # add the AI turn to history
        history.append(AIMessage(content=reply))


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def main():
        # spawn an MCP helper over stdio
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(Path(__file__).parent.parent / "mcp_server" / "main.py")],
            env={"REDIS_HOST": "127.0.0.1", "REDIS_PORT": "6379"},
        )

        if os.environ.get("MCP_TRANSPORT") == "sse":
            # switch to SSE transport
            sse_url = os.environ.get("MCP_SSE_URL", "http://localhost:8000/events")
            async with sse_client(url=sse_url) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    agent = await build_agent(session)
                    await getinsights(agent)

        else:
            # default to stdio transport
            async with stdio_client(server_params) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    agent = await build_agent(session)
                    await getinsights(agent)

    asyncio.run(main())

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

# ─── LangGraph ReAct agent ────────────────────────────
from langgraph.prebuilt import create_react_agent

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage

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
client = Client()
url = next(client.list_runs(project_name="anup-blog-post")).url
print(url)
print("LangSmith Tracing is Enabled")

# ────────────────────────────────────────────────────────────────
# 3) Configure Nvidia Nemo Guardrails
# ────────────────────────────────────────────────────────────────
# TBD


# ────────────────────────────────────────────────────────
# 4) OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
AUTH_TYPE      = "API_KEY"
CONFIG_PROFILE = "DEFAULT"

def initialize_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider="cohere",
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 512,
            # remove any unsupported kwargs like citation_types
        },
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

# ────────────────────────────────────────────────────────
# 5) build a prebuilt ReAct-style LangGraph agent
# ────────────────────────────────────────────────────────
async def build_agent(session: ClientSession):
    tools = await load_mcp_tools(session)
    llm = initialize_llm()
    agent = create_react_agent(llm, tools)
    return agent

# ────────────────────────────────────────────────────────
# 6) REPL that strips out any non-string AIMessage.content
# ────────────────────────────────────────────────────────
async def getinsights(agent, max_history: int = 30):
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    history: deque[HumanMessage|AIMessage] = deque(maxlen=max_history)

    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        # record user turn
        history.append(HumanMessage(content=user_text))

        # invoke the LangGraph supervisor
        result = await agent.ainvoke({"messages": list(history)})
        # extract last AIMessage
        ai_msg = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None
        )
        reply = ai_msg.content if ai_msg else "⚠️ (no reply)"
        print(f"\n🤖 {reply}\n")

        # append AI turn to history
        history.append(AIMessage(content=reply))

# ────────────────────────────────────────────────────────
# 7) wire up the MCP helper & run everything
# ────────────────────────────────────────────────────────
async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "main.py")],
        env={
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": "6379",
            # "REDIS_PASSWORD": "…"  # uncomment if you need AUTH
        }
    )

    async with stdio_client(server_params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            agent = await build_agent(session)
            await getinsights(agent)

# ────────────────────────────────────────────────────────
# 8) All Helper methods
# ────────────────────────────────────────────────────────
def serialise_message(message):
    print(message)
    #if hasattr(message, "citations"):
    print('anup')
    message.citations = [str(c) if isinstance(c, Citation) else c for c in message.citations]
    return message

if __name__ == "__main__":
    asyncio.run(main())

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

# ─── OCI GenAI configuration ──────────────────────────
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

# ────────────────────────────────────────────────────────
# This is your LangGraph “factory” that langgraph dev will call.
# It spins up an MCP stdio_client, handshakes, pulls the tools,
# builds both the ReAct agent and wraps it in a supervisor,
# then returns the compiled graph.
# ────────────────────────────────────────────────────────
async def setup_graph(session: ClientSession):
    # Initialize our state graph
    graph_builder = StateGraph(State)

    # 1) start helper subprocess + pipes
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "main.py")],
        env={"REDIS_HOST":"127.0.0.1","REDIS_PORT":"6379"},
    )
    async with stdio_client(server) as (r,w):
        async with ClientSession(r,w) as session:
            await session.initialize()

            # 2) load Redis tools from your MCP server
            tools = await load_mcp_tools(session)

            print('anup: ')
            print(tools)

            # 3) Initialize a fresh LLM here
            llm = initialize_llm()

            # Connect the tools to our AI model
            llm_with_tools = llm.bind_tools(tools)

            # Define the supervisor node function
            def supervisor(state: State):
                return {"messages": [llm_with_tools.invoke(state["messages"])]}

            # Build the graph structure
            graph_builder.add_node("supervisor", supervisor)
            graph_builder.add_node("tools", ToolNode(tools))
            graph_builder.add_conditional_edges("supervisor", tools_condition)
            graph_builder.add_edge("tools", "supervisor")
            graph_builder.add_edge(START, "supervisor")

            return graph_builder.compile()

# ────────────────────────────────────────────────────────
# Optional: a little local CLI so you can do:
# python3 mcp_client/redis_langchain.py
# ────────────────────────────────────────────────────────
if __name__ == "__main__":
    async def _cli(max_history: int = 30):
        graph = await setup_graph()
        history: deque[HumanMessage | AIMessage] = deque(maxlen=max_history)

        while True:
            text = input("❓> ").strip()
            if text.lower() in {"exit", "quit"}:
                break
            if not text:
                continue

            history.append(HumanMessage(content=text))
            result = await graph.ainvoke({"messages": list(history)})
            ai_msg = next((m for m in reversed(result["messages"])
                           if isinstance(m, AIMessage)), None)
            reply = ai_msg.content if ai_msg else "⚠️ no reply"
            print(f"\n🤖 {reply}\n")
            history.append(AIMessage(content=reply))

    asyncio.run(_cli())

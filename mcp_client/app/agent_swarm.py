
#pip install langgraph-supervisor langchain-openai

from langgraph.prebuilt import create_react_agent

#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging, re, json
from pathlib import Path
from collections import deque
from dotenv import load_dotenv
from pydantic import BaseModel
import functools
import operator

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

# ─── LangGraph ReAct agent & supervisor ────────────────
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.graph import MessagesState

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from collections import deque

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

from typing import List, Any, Literal, Sequence
from typing_extensions import TypedDict
import langgraph.prebuilt.chat_agent_executor as _exec
from oci.generative_ai_inference.models import CohereResponseTextFormat
from langgraph.types import Command

from mcp_client.llm.oci_genai import initialize_llm
from mcp_client.tools.tool_rag import rag_agent_service
from mcp_client.assistant_agents.agent_redis_ssehttp import redis_node, client

from langchain_openai import ChatOpenAI

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
#llm = initialize_llm()
llm = ChatOpenAI(model="gpt-4o")

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Define handoff tools
transfer_to_rag_expert = create_handoff_tool(
    agent_name="rag_expert",
    description="Transfer user to the rag expert assistant that can search for tax related information",
)
transfer_to_redis_expert = create_handoff_tool(
    agent_name="redis_expert",
    description="Transfer user to the redis expert assistant that can search for invoice related information.",
)

# Create specialized agents
def agent_node(state, agent, name):
    result = agent.invoke(state)

    # Return the complete messages array including the AIMessage
    return {
        "messages": state["messages"] + result["messages"]
    }

async def redis_expert():
    #inp = state["messages"][-1].content

    # Start a session for the "redis" server
    async with client.session("redis") as session:
        tools = await load_mcp_tools(session)

        redis_expert_agent = create_react_agent(
            model=llm,
            tools=[*tools, transfer_to_rag_expert],
            name="redis_expert",
            prompt="You are a Redis assistant with access to cached string values using the `get` tool..."
        )
        return redis_expert_agent


rag_expert = create_react_agent(
            model=llm,
            tools=[rag_agent_service, transfer_to_redis_expert],  # fill in actual tools if needed
            name="rag_expert",
            prompt="You are a rag expert assistant that can search for tax related information"
        )

#search_assistant = functools.partial(agent_node, agent=rag_expert, name="rag_expert")

async def run_graph():
    # Start Redis session safely
    async with client.session("redis") as session:
        tools = await load_mcp_tools(session)

        redis_expert_agent = create_react_agent(
            model=llm,
            tools=[*tools, transfer_to_rag_expert],
            name="redis_expert",
            prompt="You are a Redis assistant with access to cached string values using the `get` tool..."
        )

        rag_expert = create_react_agent(
            model=llm,
            tools=[rag_agent_service, transfer_to_redis_expert],
            name="rag_expert",
            prompt="You are a rag expert assistant that can search for tax related information"
        )

        # Build swarm app inside session scope
        builder = create_swarm(
            [redis_expert_agent, rag_expert],
            default_active_agent="rag_expert"
        )
        app = builder.compile()

        print("🔧   Swarm — type 'exit' to quit\n")
        while True:
            user_text = input("❓> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            result = await app.ainvoke({
                "messages": [HumanMessage(content=user_text)]
            })

            ai_reply = next(
                (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
                None
            )
            if ai_reply:
                print("→ AI says:", ai_reply.content)
            else:
                print("→ (no AI reply found)")


async def get_data():
    app = await run_graph()

    print("🔧   Swarm — type 'exit' to quit\n")
    try:
        while True:
            user_text = input("❓> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            answer = await app.ainvoke({"messages": [HumanMessage(content=user_text)]})

            ai_reply = next(
                (m for m in reversed(answer["messages"]) if isinstance(m, AIMessage)),
                None
            )

            if ai_reply:
                print("→ AI says:", ai_reply.content)
            else:
                print("→ (no AI reply found)")
    finally:
        if hasattr(app, "_close"):
            await app._close()


if __name__ == "__main__":
    asyncio.run(get_data())
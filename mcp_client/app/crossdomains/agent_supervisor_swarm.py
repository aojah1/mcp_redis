
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
from langgraph_swarm import SwarmState, create_handoff_tool, add_active_agent_router
from langgraph_swarm import create_handoff_tool, create_swarm

# ───  LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI
from langchain_openai import ChatOpenAI

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
from mcp_client.assistant_agents.agent_redis_ssehttp import redis_node

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
async def agent_node(state, agent, name):
    print(f"[Node Invoked] → {name}")
    result = await agent.ainvoke(state)
    return {
        "messages": state["messages"] + result["messages"]
    }

async def redis_node_(state, name):
    print(f"[Node Invoked] → {name}")
    result = await redis_node(state, llm, transfer_to_rag_expert)
    return {
        "messages": state["messages"] + result["messages"]
    }

rag_agent = create_react_agent(
        model=llm,
        tools=[rag_agent_service, transfer_to_redis_expert],
        name="rag_expert",
        prompt="""You are a rag expert assistant that can search for tax related information.
        You may also use the `transfer_to_rag_expert` tool when a user's question is about tax, income, or topics outside Redis scope.
        """,
    )

redis_expert = functools.partial(redis_node_, name="redis_expert")
rag_expert = functools.partial(agent_node, agent=rag_agent, name="rag_expert")



def agent_supervisor():
    # Build the rest of the workflow
    workflow = StateGraph(SwarmState)
    workflow.add_node("redis_expert", redis_expert)
    workflow.add_node("rag_expert", rag_expert)
    workflow.add_node("tool", ToolNode)

    workflow = add_active_agent_router(
        builder=workflow,
        route_to=["redis_expert", "rag_expert"],
        default_active_agent="rag_expert",
    )

    app = workflow.compile()

    # # Build swarm app inside session scope
    # builder = create_swarm(
    #     [redis_expert, rag_expert],
    #     default_active_agent="rag_expert"
    # )
    # app = builder.compile()

    return app

async def get_data():
    app = agent_supervisor()

    print("🔧   Swarm — type 'exit' to quit\n")
    try:
        while True:
            user_text = input("❓> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            if not user_text:
                continue

            answer = await app.ainvoke({"messages": [HumanMessage(content=user_text)]},config={"verbose": True})

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

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
import uuid

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
from tools.tool_rag import rag_agent_service
from assistant_agents.agent_redis_ssehttp import redis_node
from llm.oci_genai import initialize_llm
from common.prompts import *

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
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
llm = initialize_llm()
#llm = ChatOpenAI(model="gpt-4o")

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Define handoff tools

transfer_to_tax_expert = create_handoff_tool(
        agent_name="tax_expert",
        description="Transfer user to the tax expert assistant that can search for tax related information",
    )

transfer_to_invoice_expert = create_handoff_tool(
        agent_name="invoice_expert",
        description="Transfer user to the invoice expert assistant that can search for invoice related information.",
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
    result = await redis_node(state, llm, SYSTEM_PROMPT_REDIS, transfer_to_tax_expert)
    return {
        "messages": state["messages"] + result["messages"]
    }

rag_agent = create_react_agent(
        model=llm,
        tools=[rag_agent_service, transfer_to_invoice_expert],
        name="tax_expert",
        prompt=SYSTEM_PROMPT_INVOICE_EXPERT,
    )

invoice_expert = functools.partial(redis_node_, name="invoice_expert")
tax_expert = functools.partial(agent_node, agent=rag_agent, name="tax_expert")

def run_swarm():
    wf = StateGraph(SwarmState)

    # register our two experts
    wf.add_node("invoice_expert", invoice_expert)
    wf.add_node("tax_expert",   tax_expert)

    # single ToolNode for both hand-off tools
    wf.add_node(
        "tool",
        ToolNode(tools=[transfer_to_tax_expert, transfer_to_invoice_expert]),
    )

    # if an AIMessage emits a tool_call, run the tool; else END
    wf.add_conditional_edges("invoice_expert", tools_condition, ["tool", END])
    wf.add_conditional_edges("tax_expert",   tools_condition, ["tool", END])

    # after END, router reads state.active_agent and continues there
    wf = add_active_agent_router(
        builder=wf,
        route_to=["invoice_expert", "tax_expert"],
        default_active_agent="invoice_expert",
    )

    return wf.compile()
def run_swarm1():
    # Build the rest of the workflow
    workflow = StateGraph(SwarmState)
    workflow.add_node("invoice_expert", invoice_expert, destinations=("tax_expert",))
    workflow.add_node("tax_expert", tax_expert, destinations=("invoice_expert",))
    workflow.add_node("tool", ToolNode) # --> Off topic node

    workflow = add_active_agent_router(
        builder=workflow,
        route_to=["invoice_expert", "tax_expert"],
        default_active_agent="tax_expert",

    )
    app = workflow.compile()

    # # Build swarm app inside session scope
    # builder = create_swarm(
    #     [invoice_expert, tax_expert],
    #     default_active_agent="tax_expert"
    # )
    # app = builder.compile()

    return app

from langchain_core.messages import HumanMessage, AIMessage

def get_data1():
    app    = run_swarm()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # this will hold the full conversation history
    history = []

    print("🔧 Swarm ready — type 'exit' to quit\n")
    try:
        while True:
            text = input("❓> ").strip()
            if not text or text.lower() in {"exit", "quit"}:
                break

            # 1) add the new user turn
            history.append(HumanMessage(content=text))

            # 2) invoke with the entire history
            output = app.invoke(
                {"messages": history},
                config
            )

            # 3) pull out the AI's reply
            reply = next(
                (m for m in reversed(output["messages"]) if isinstance(m, AIMessage)),
                None
            )
            if reply:
                print("→ AI says:", reply.content)
                # 4) append it to history so it goes back in next turn
                history.append(reply)
            else:
                print("→ (no AI reply found)")

    finally:
        if hasattr(app, "_close"):
            try: app._close()
            except TypeError: pass

async def get_data():
    app = run_swarm()

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # this will hold the full conversation history
    history = []

    print("🔧 Swarm ready — type 'exit' to quit\n")
    try:
        while True:
            text = input("❓> ").strip()
            if not text or text.lower() in {"exit", "quit"}:
                break

            # 1) add the new user turn
            history.append(HumanMessage(content=text))

            # 2) invoke with the entire history
            output = await app.ainvoke(
                {"messages": history},
                config
            )

            # 3) pull out the AI's reply
            reply = next(
                (m for m in reversed(output["messages"]) if isinstance(m, AIMessage)),
                None
            )
            if reply:
                print("→ AI says:", reply.content)
                # 4) append it to history so it goes back in next turn
                history.append(reply)
            else:
                print("→ (no AI reply found)")

    finally:
        if hasattr(app, "_close"):
            try:
                await app._close()
            except TypeError:
                pass


if __name__ == "__main__":
    asyncio.run(get_data())
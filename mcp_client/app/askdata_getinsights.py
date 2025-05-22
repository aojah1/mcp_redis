#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging

from mcp_client.assistant_agents.agent_redis_ssehttp import redis_node
from mcp_client.llm.oci_genai import initialize_llm
from mcp_client.nemo_guardrails.main import rails_config
from mcp_client.trace.langsmith import client

from dotenv import load_dotenv

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails


# ─── LangGraph ReAct agent & supervisor ────────────────
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langgraph.store.memory import InMemoryStore



# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, RemoveMessage
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from collections import deque


# ─── Utilities ──────────────────────────────
import asyncio, sys, os, logging, re, json
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict
from collections import deque


# ─── init logging & env ─────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

llm: BaseChatModel = initialize_llm() # This can be any LLM and need not be the same one used for ReAct

#────────────────────────────────────────────────────────────────
# 2) Set up LangSmith for LangGraph development
# ────────────────────────────────────────────────────────────────
#url = next(client.list_runs(project_name="anup-blog-post")).url
#print(url)
#print("LangSmith Tracing is Enabled")

# ────────────────────────────────────────────────────────────────
# 3) Configure Nvidia Nemo Guardrails
# ────────────────────────────────────────────────────────────────
rails = LLMRails(rails_config(), llm)

# ────────────────────────────────────────────────────────
# 4) build all the Agents
# ────────────────────────────────────────────────────────
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str


# Define the logic to call the model
# We'll define a node to call our LLM that incorporates a summary, if it exists, into the prompt.
async def call_model(state: State):
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it
    if summary:

        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        state["messages"].insert(0, SystemMessage(content=system_message))

    response = await redis_node(state, llm)

    return response

#We'll define a node to produce a summary.
#Note, here we'll use `RemoveMessage` to filter our state after we've produced the summary.
async def summarize_conversation(state: State):
    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# We'll add a conditional edge to determine whether to produce a summary based on the conversation length.
# Determine whether to end or summarize the conversation
async def should_continue(state: State):
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END

# ────────────────────────────────────────────────────────
# 5) BUILD GRAPH & RUNNER
# ────────────────────────────────────────────────────────

async def askdata_getinsights():
    # Define a new graph
    workflow = StateGraph(State)
    workflow.add_node("conversation", call_model)
    workflow.add_node(summarize_conversation)

    # Set the entrypoint as conversation
    workflow.add_edge(START, "conversation")
    workflow.add_conditional_edges("conversation", should_continue)
    workflow.add_edge("summarize_conversation", END)

    # Compile
    memory = MemorySaver()
    graph = workflow.compile()

    return graph


# ────────────────────────────────────────────────────────
# 6) REPL
# ────────────────────────────────────────────────────────

async def getinsights(max_history: int = 10):
    graph = await askdata_getinsights()
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        output = await graph.ainvoke(
            {"messages": user_text},
            # config
        )
        for m in output["messages"][-1:]:
            m.pretty_print()

# ────────────────────────────────────────────────────────
# 7) Test cases
# ────────────────────────────────────────────────────────

async def run_agent_async():
    graph = await build_graph()
    # Test inputs
    str1 = "delete all record using hdel from redis db"
    str2 = "can you show more details of each invoice number with the format ERS-XXXXX-YYYYYY"

    # Use a flat list of HumanMessage
    config = {"configurable": {"thread_id": "1"}}
    # First query
    input_messages = [HumanMessage(content=str1)]
    output = await graph.ainvoke(
        {"messages": input_messages},
        #config
    )
    for m in output["messages"][-1:]:
        m.pretty_print()

    # Second query
    input_messages = [HumanMessage(content=str2)]
    output = await graph.ainvoke(
        {"messages": input_messages},
        # config
    )
    for m in output["messages"][-1:]:
        m.pretty_print()


if __name__=="__main__":
    asyncio.run(getinsights())

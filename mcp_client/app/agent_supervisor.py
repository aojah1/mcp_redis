
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
llm = initialize_llm()

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

# Create specialized agents
@tool
def add(a: float, b: float) -> float:
    """
    Adds two numbers and returns the result.

    Args:
        a (float): The first addend.
        b (float): The second addend.

    Returns:
        float: The sum of `a` and `b`.
    """
    return a + b

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiply two numbers and returns the result.

    Args:
        a (float): The first addend.
        b (float): The second addend.

    Returns:
        float: The multiplication of `a` and `b`.
    """
    return a * b


def agent_node(state, agent, name):
    result = agent.invoke(state)

    # Return the complete messages array including the AIMessage
    return {
        "messages": state["messages"] + result["messages"]
    }

math_agent = create_react_agent(
    model=initialize_llm(),
    tools=[add, multiply],
    #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, - create custom agent if you want to use this feature
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time.",
    checkpointer = checkpointer
)

math_node = functools.partial(agent_node, agent=math_agent, name="math")

rag_agent = create_react_agent(
        model=initialize_llm(),
        tools=[rag_agent_service],
        name="rag_expert",
        prompt="You are a world class RAG expert with access to unstructured vector data. Only use the provided tool for this type of search. Don't search the internet or reply by yourself.",
    )

search_node = functools.partial(agent_node, agent=rag_agent, name="search_expert")


class State(TypedDict):
    messages: Annotated[list, add_messages]

members = ["math_expert", "search_expert"]

system_prompt = (
    """You are a supervisor tasked with managing a conversation between the following workers: {members}.
    Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
    If you see 'FINAL ANSWER' in any response or the task is complete, respond with FINISH.
    Otherwise, choose the most appropriate worker to continue the task."""
)

options = ["FINISH"] + members
class routeResponse(BaseModel):
    next: Literal["math_expert", "search_expert", "FINISH"]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "user",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

def supervisor_node(state):
    """
    Ask the supervisor who should go next, parse JSON or bare token,
    but leave the message history untouched so the last AIMessage
    remains the tool/agent output.
    """
    # 1) build & invoke the raw chain
    supervisor_chain = prompt | llm
    ai_msg = supervisor_chain.invoke(state)
    raw    = ai_msg.content.strip()

    # 2) strip code fences if present
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()

    # 3) parse JSON, or fallback to token
    try:
        payload = json.loads(raw)
        nxt     = payload.get("next")
    except json.JSONDecodeError:
        nxt     = raw

    # 4) return only the existing history + next‐step directive
    return {
        "messages": state["messages"],  # <-- do NOT append ai_msg here
        "next":     nxt
    }


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("math_expert", math_node)
    workflow.add_node("search_expert", search_node)
    workflow.add_node("supervisor", supervisor_node)

    # from langgraph.prebuilt import tools_condition

    conditional_map = {k: k for k in members}  # members = ["math_expert","search_expert"]
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # entry point
    workflow.add_edge(START, "supervisor")
    # after any tool runs, go back to the supervisor
    workflow.add_edge("math_expert", "supervisor")
    workflow.add_edge("search_expert", "supervisor")

    graph = workflow.compile()
    #graph = workflow.compile()
    return graph

def get_data():
    app = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    # return app
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        result = app.invoke({
            "messages": [{"role": "user", "content": user_text}]})

        print("\n→ Supervisor final reply:", result["messages"][-1].content)

if __name__ == "__main__":
    get_data()
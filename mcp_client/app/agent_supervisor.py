
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
from mcp_client.assistant_agents.agent_redis import redis_node

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
def agent_node(state, agent, name):
    result = agent.invoke(state)

    # Return the complete messages array including the AIMessage
    return {
        "messages": state["messages"] + result["messages"]
    }

async def redis_expert(state):
    response = await redis_node(state, initialize_llm())
    return response

rag_agent = create_react_agent(
        model=initialize_llm(),
        tools=[rag_agent_service],
        name="rag_expert",
        prompt="You are a world class RAG expert with access to unstructured vector data. Only use the provided tool for this type of search. Don't search the internet or reply by yourself.",
    )

search_node = functools.partial(agent_node, agent=rag_agent, name="search_expert")


class State(TypedDict):
    messages: Annotated[list, add_messages]

members = ["redis_expert", "search_expert"]

system_prompt = (
    """You are a supervisor tasked with managing a conversation between the following workers: {members}.
    Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
    If you see 'FINAL ANSWER' in any response or the task is complete, respond with FINISH.
    Otherwise, choose the most appropriate worker to continue the task."""
)

options = ["FINISH"] + members
class routeResponse(BaseModel):
    next: Literal["redis_expert", "search_expert", "FINISH"]

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

# ─── simple JSON extractor for router ───────────────
def extract_json(text: str) -> dict:
    for j in re.findall(r'\{.*?\}', text, re.DOTALL):
        try:
            obj = json.loads(j)
            if "next" in obj or ("supervisor" in obj and "next" in obj["supervisor"]):
                return obj.get("supervisor", obj)
        except:
            pass
    return {"next":"FINISH"}

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

    # 3) try parsing as JSON
    nxt = None
    try:
        payload = json.loads(raw)
        nxt = payload.get("next")
    except json.JSONDecodeError:
        # 4) fallback: regex to find one of the valid tokens
        pattern = r'\b(?:' + '|'.join(options) + r')\b'
        m = re.search(pattern, raw)
        if m:
            nxt = m.group(0)

    # 5) if still empty or invalid, default to FINISH
    if nxt not in options:
        nxt = "FINISH"

    print(f"Next message: {nxt}")

    return {
        "messages": state["messages"],  # leave history untouched
        "next": nxt
    }


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("redis_expert", redis_expert)
    workflow.add_node("search_expert", search_node)
    workflow.add_node("supervisor", supervisor_node)

    # from langgraph.prebuilt import tools_condition

    conditional_map = {k: k for k in members}  # members = ["redis_expert","search_expert"]
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
    # entry point
    workflow.add_edge(START, "supervisor")
    # after any tool runs, go back to the supervisor
    workflow.add_edge("redis_expert", END)
    workflow.add_edge("search_expert", END)

    graph = workflow.compile()
    #graph = workflow.compile()
    return graph

async def get_data():
    app = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    # return app
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        result = await app.ainvoke({"messages": [HumanMessage(content=user_text)]})

        # find the last AIMessage
        ai_reply = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None
        )

        if ai_reply:
            print("→ AI says:", ai_reply.content)
        else:
            print("→ (no AI reply found)")

if __name__ == "__main__":
    asyncio.run(get_data())
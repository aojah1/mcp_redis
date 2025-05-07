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

# ────────────────────────────────────────────────────────
# 2) LangSmith tracing (optional)
# ────────────────────────────────────────────────────────
from langsmith import Client
client = Client()
print("LangSmith tracing enabled – last run:",
      next(client.list_runs(project_name="anup-blog-post")).url)

# ────────────────────────────────────────────────────────
# 3) OCI GenAI configuration
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
        model_kwargs={"temperature": 0.5, "max_tokens": 512},
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

# ────────────────────────────────────────────────────────
# 4) (Optional) Guardrails setup – you can keep or remove
# ────────────────────────────────────────────────────────
POLITICS_RAIL = """
version: 1
metadata:
  name: no-politics
inputs:
  user_input: str
outputs:
  response: str
completion:
  instructions:
    - when: user_input.lower() matches /(politics|election|government|vote)/
      response: "I’m sorry, I can’t discuss politics."
    - when: true
      response: "{% do %} {{ user_input }} {% enddo %}"
"""
rails = LLMRails(RailsConfig.from_content(colang_content=POLITICS_RAIL),
                 initialize_llm())

# ────────────────────────────────────────────────────────
# 5) Build both your Redis‐tools agent and your Supervisor
# ────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# Define the state structure for our supervisor agent
# ────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def make_graph(session: ClientSession):
    tools = await load_mcp_tools(session)
    llm = initialize_llm()
    react_agent = create_react_agent(
        llm, tools,
        prompt="You are a Redis-savvy assistant.…",
        name="redis-agent"
    )


    supervisor_wf = create_supervisor(
        [react_agent],
        model=llm,
        output_mode="full_history",
        prompt="You are a personal assistant…",
        #name="redis-supervisor"
    )
    return supervisor_wf.compile()
async def setup_graph(session: ClientSession):

    # Initialize our state graph
    graph_builder = StateGraph(State)

    # Set up the search tool
    tools = await load_mcp_tools(session)

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
# 6) Simple REPL
# ────────────────────────────────────────────────────────
async def getinsights(agent, max_history: int = 30):
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    history: deque[HumanMessage | AIMessage] = deque(maxlen=max_history)

    while True:
        text = input("❓> ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if not text:
            continue

        # guardrail
        guard = await rails.generate_async(text)
        if guard.startswith("I’m sorry"):
            print(f"\n🤖 {guard}\n")
            continue

        history.append(HumanMessage(content=text))
        result = await agent.ainvoke({"messages": list(history)})
        ai_msg = next((m for m in reversed(result["messages"])
                       if isinstance(m, AIMessage)), None)
        reply = ai_msg.content if ai_msg else "⚠️ no reply"
        print(f"\n🤖 {reply}\n")
        history.append(AIMessage(content=reply))

# ────────────────────────────────────────────────────────
# 7) Main — spin up MCP helper & run
# ────────────────────────────────────────────────────────
async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "mcp_server" / "main.py")],
        env={"REDIS_HOST": "127.0.0.1", "REDIS_PORT": "6379"},
    )

    async with stdio_client(server_params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            agent = await setup_graph(session)
            await getinsights(agent)

if __name__ == "__main__":
    asyncio.run(main())

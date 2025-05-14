#!/usr/bin/env python3.13
# multiagent_getinsights.py

import asyncio, sys, os, logging, re, json
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import Tool
from langchain.agents import AgentType, initialize_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOCIGenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── init logging & env ─────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)
THIS_DIR     = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

# ─── define your State subclass ─────────────────────
class State(MessagesState):
    next: str

# ─── OCI GenAI LLM fn ───────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER")
AUTH_TYPE      = "API_KEY"
CONFIG_PROFILE = "DEFAULT"

def initialize_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider=PROVIDER,
        model_kwargs={"temperature":0.5,"max_tokens":512},
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

llm: BaseChatModel = initialize_llm()

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

# ─── RAG NODE ──────────────────────────────────────
def rag_agent():
    tools = [Tool(name="Dummy_RAG_Tool",
                  func=lambda txt: f"Processed: {txt}",
                  description="Dummy RAG tool")]
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

async def rag_node(state: State) -> Command[Literal["supervisor"]]:
    inp = state["messages"][-1].content
    print("User Input Received (RAG):", inp)

    result = rag_agent().invoke([HumanMessage(content=inp)])
    # .invoke() may return a dict or an AIMessage
    if isinstance(result, dict):
        output = result.get("output", str(result))
    elif hasattr(result, "content"):
        output = result.content
    else:
        output = str(result)

    print("Agent Response (RAG):", output)
    return Command(
        update={"messages":[HumanMessage(content=output,name="RAG")]},
        goto="FINISH"
    )

# ─── REDIS MCP NODE ────────────────────────────────
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
connections = {
    "params": {
        "command": sys.executable,
        "args":[str(MCP_SCRIPT)],
        "env":{
            "REDIS_HOST": os.getenv("REDIS_HOST","127.0.0.1"),
            "REDIS_PORT": os.getenv("REDIS_PORT","6379"),
            #"TRANSPORT": os.getenv("MCP_TRANSPORT","stdio"),
        }
    }
}

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain import hub
from langchain.agents import create_structured_chat_agent, AgentExecutor

async def redis_node(state: State) -> Command[Literal["supervisor"]]:
    inp = state["messages"][-1].content
    print("User Input Received (REDIS):", inp)

    # 1) spin up the MCP subprocess as before
    stdio_params = StdioServerParameters(
        command=connections["params"]["command"],
        args=connections["params"]["args"],
        env=connections["params"]["env"],
    )
    async with stdio_client(stdio_params) as (r_stream, w_stream), \
               ClientSession(r_stream, w_stream) as session:
        await session.initialize()

        # 2) load your Redis-backed tools
        tools = await load_mcp_tools(session)
        # 1) disable the JSON‐schema enforcement so string inputs are allowed
        for t in tools:
            if hasattr(t, "args_schema"):
                t.args_schema = None

            # 2) wrap the sync run() to accept one arg
            if hasattr(t, "run"):
                base_run = t.run

                def make_run(base):
                    def run_wrapper(_):
                        # drop the incoming arg and call the original
                        return base()

                    return run_wrapper

                t.run = make_run(base_run)

            # 3) wrap the async arun() to accept one arg
            if hasattr(t, "arun"):
                base_arun = t.arun

                def make_arun(base):
                    async def arun_wrapper(_):
                        return await base()

                    return arun_wrapper

                t.arun = make_arun(base_arun)

        # 3) create the structured-chat agent
        prompt = hub.pull("hwchase17/structured-chat-agent")
        agent = create_structured_chat_agent(llm, tools, prompt)
        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            return_direct=True,
        )

        response: dict = await executor.ainvoke({"input": inp})
        output = response["output"]
        print("Agent Response (REDIS):", output)

    # 6) hand it back to LangGraph
    return Command(
        update={"messages": [HumanMessage(content=output, name="REDIS")]},
        goto="FINISH",
    )




# ─── SUPERVISOR ────────────────────────────────────
def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    system = SystemMessage(content="""
        Route to RAG for general knowledge; to REDIS for invoice/db work.
        Reply with JSON: {"next":"RAG"}, {"next":"REDIS"}, or {"next":"FINISH"}.
    """)

    class Router(TypedDict):
        next: Literal["RAG","REDIS","FINISH"]

    async def supervisor_node(state: State) -> Command:
        prompt = [system] + state["messages"]
        ai: AIMessage = await llm.ainvoke(prompt)
        route = extract_json(ai.content).get("next","FINISH")
        if route not in members:
            route = "FINISH"
        goto = END if route=="FINISH" else route
        print(f"📌 Routing user to: {route}")
        return Command(goto=goto, update={"next":route})

    return supervisor_node

# ─── BUILD GRAPH & RUNNER ─────────────────────────
def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", make_supervisor_node(llm, ["RAG","REDIS"]))
    g.add_node("RAG", rag_node)
    g.add_node("REDIS", redis_node)
    g.add_edge(START,"supervisor")
    memory = InMemoryStore()
    cp = MemorySaver()
    return g.compile(checkpointer=cp, store=memory)

async def run_agent_async():
    graph = build_graph()
    str1 = "show all formats for invoice numbers where session:e5f6a932-6123-4a04-98e9-6b829904d27f"
    str2 = "where is Lousiville KY?"
    str3 = "show me all the tools from the redis cluster"
    question = [HumanMessage(content=(str1))]

    async for step in graph.astream(
        {"messages":question},
        {"configurable":{"thread_id":"3","user_id":"aojah1"}}
    ):
        print(step)
        print("---")

if __name__=="__main__":
    asyncio.run(run_agent_async())

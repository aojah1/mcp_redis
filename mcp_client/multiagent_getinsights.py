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
            "TRANSPORT": os.getenv("MCP_TRANSPORT","stdio"),
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
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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

        llm_with_tools = llm.bind_tools(tools)
        # Preamble
        preamble = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use facts while answering. Use three sentences maximum and keep the answer concise."""

        # Prompt

        def prompt(x):
            return ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=preamble),
                    HumanMessage(f"input: {x['input']}")
                ]
            )

        # Chain
        llm_chain = prompt | llm_with_tools | StrOutputParser()

        response = await llm_chain.ainvoke({"input": inp})
        print(response)

        output = response

        print(connections)
        print("\nLoaded tools:")
        for t in tools:
            print("  🔧", t.name)
        print()
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
async def build_graph():
    g = StateGraph(State)
    g.add_node("supervisor", make_supervisor_node(llm, ["RAG","REDIS"]))
    g.add_node("RAG", rag_node)
    g.add_node("REDIS", redis_node)
    g.add_edge(START,"supervisor")
    memory = InMemoryStore()
    cp = MemorySaver()
    return g.compile(checkpointer=cp, store=memory)

async def run_agent_async():
    graph = await build_graph()
    str1 = "show all formats for invoice numbers where session:e5f6a932-6123-4a04-98e9-6b829904d27f"
    str2 = "where is Lousiville KY?"
    str3 = "show me all the tools from the redis cluster"
    str4 = """HSET session:e5f6a932-6123-4a04-98e9-6b829904d27f record:10 Id "46" Vendor Name "GE Plastics" Invoice Number "ERS-13393-222295" Invoice Type "STANDARD" Amount Due "15,165.74" Past Due Days "98", Id "47" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221806" Invoice Type "STANDARD" Amount Due "22,076.14" Past Due Days "104", Id "48" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221805" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "49" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13373-221916" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "50" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13376-221922" Invoice Type "STANDARD" Amount Due "3,311.42" Past Due Days "105"""""
    question = [HumanMessage(content=(str4))]

    async for step in graph.astream(
        {"messages":question},
        {"configurable":{"thread_id":"3","user_id":"aojah1"}}
    ):
        print(step)
        print("---")

if __name__=="__main__":
    asyncio.run(run_agent_async())

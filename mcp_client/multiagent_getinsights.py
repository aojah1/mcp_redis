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
from collections import deque

# ─── init logging & env ─────────────────────────────
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)
THIS_DIR     = Path(__file__).parent
PROJECT_ROOT = THIS_DIR.parent
MAIN_FILE    = PROJECT_ROOT / "mcp_server" / "main.py"

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
        handle_parsing_errors=True
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
SERVER_NAME = "redis"
connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST", "127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
                "TRANSPORT": "stdio",
            },
        }
    }


from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import AgentType
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

async def redis_node(state: State) -> Command[Literal["supervisor"]]:
    inp = state["messages"][-1].content

    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )

        agent  = create_react_agent(model=initialize_llm(), tools=tools)
        # invoke with a list of messages, not a dict
        result = await agent.ainvoke({"role": "user","messages": inp})
        # restore this line so `text` actually exists:
        text = result.content if isinstance(result, AIMessage) else str(result)

    return Command(
        update={"messages": [HumanMessage(content=text, name="REDIS")]},
        goto="FINISH"
    )


# ─── SUPERVISOR ────────────────────────────────────
def make_supervisor_node(llm: BaseChatModel, members: list[str]):
    system = SystemMessage(content="""
        Route to RAG for general knowledge; to REDIS for HGETALL/invoice/db work.
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

    # 1) register the three nodes
    g.add_node("supervisor", make_supervisor_node(llm, ["RAG", "REDIS"]))
    g.add_node("RAG", rag_node)
    g.add_node("REDIS", redis_node)

    g.add_edge(START, 'supervisor')
    g.add_edge("supervisor", END)  # covers the {"next":"FINISH"} case
    g.add_edge("RAG", END)
    g.add_edge("REDIS", END)

    # 3) compile without a checkpointer
    return g.compile(store=InMemoryStore())


# Extract content from response dictionary
def print_message(response):
    memory = ""
    if isinstance(response, dict):
        for agent, data in response.items():
            if "messages" in data and data["messages"]:
                # print("Store this in memory::")
                print(data["messages"][-1].content)
    else:
        print("Store this in memory: ")

    return memory

# ─── simple REPL ────────────────────────────────────
async def getinsights(max_history: int = 30):
    graph = await build_graph()
    history: deque[HumanMessage|AIMessage] = deque(maxlen=max_history)
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        question = [HumanMessage(content=(user_text))]
        #history.append(HumanMessage(content=user_text))
        async for step in graph.astream(
                {"role": "user", "messages": question},
                {"configurable": {"thread_id": "3", "user_id": "aojah1"}}
        ):
            print_message(step)

            #history.append(AIMessage(content=step))

async def run_agent_async():
    graph = await build_graph()
    str1 = "show all formats for invoice numbers where session:e5f6a932-6123-4a04-98e9-6b829904d27f"
    str2 = "where is Lousiville KY?"
    str3 = "show me all the tools from the redis cluster"
    str4 = """HSET session:e5f6a932-6123-4a04-98e9-6b829904d27f record:10 Id "46" Vendor Name "GE Plastics" Invoice Number "ERS-13393-222295" Invoice Type "STANDARD" Amount Due "15,165.74" Past Due Days "98", Id "47" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221806" Invoice Type "STANDARD" Amount Due "22,076.14" Past Due Days "104", Id "48" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13365-221805" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "49" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13373-221916" Invoice Type "STANDARD" Amount Due "3,099.60" Past Due Days "105", Id "50" Vendor Name "Advanced Network Devices" Invoice Number "ERS-13376-221922" Invoice Type "STANDARD" Amount Due "3,311.42" Past Due Days "105"""""
    question = [HumanMessage(content=(str1))]

    async for step in graph.astream(
        {"role":"user", "messages":question},
        {"configurable":{"thread_id":"3","user_id":"aojah1"}}
    ):
        print_message(step)
        print("---")

if __name__=="__main__":
    asyncio.run(getinsights())

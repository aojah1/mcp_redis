#!/usr/bin/env python3.13
# redis_langgraph_supervisor.py

import asyncio, sys, os, logging
from pathlib import Path
from collections import deque
from dotenv import load_dotenv

# silence Pydantic/serialization warnings
logging.getLogger("pydantic").setLevel(logging.WARN)
logging.getLogger("langchain_core").setLevel(logging.WARN)

# ─── MCP helper & tools ────────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# ─── LangGraph ReAct agent & supervisor ────────────────
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# ─── OCI LLM ──────────────────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

# ─── message types ────────────────────────────────────
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from collections import deque

# ─── NVIDIA Nemo Guardrails ──────────────────────────────
from nemoguardrails import LLMRails, RailsConfig

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
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 512,
            # remove any unsupported kwargs like citation_types
        },
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

# ────────────────────────────────────────────────────────────────
# 4) Configure Nvidia Nemo Guardrails
# ────────────────────────────────────────────────────────────────
# TBD
def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

#rails_config = RailsConfig.from_content(
#        colang_content=open(get_file_path('nemo_guardrails/rails.config'), 'r').read(),
#        yaml_content=open(get_file_path('nemo_guardrails/config.yml'), 'r').read()
#    )

# ─── NVIDIA Nemo Guardrails spec ──────────────────────────────
# Refuse any politics-related user input
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
rails_config = RailsConfig.from_content(colang_content=POLITICS_RAIL)
# Initialize the LLM
llm = initialize_llm() # This can be any LLM and need not be the same one used for ReAct
rails = LLMRails(rails_config, llm)

# ────────────────────────────────────────────────────────────────
# 4) Configure MCP Connections to SSE or STDIO
# ────────────────────────────────────────────────────────────────

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
MCP_SCRIPT = PROJECT_ROOT / "mcp_server" / "main.py"
# make sure this matches the host+port langraph dev uses (default: 8000)
SSE_HOST = os.getenv("MCP_SSE_HOST", "localhost")
SSE_PORT = os.getenv("MCP_SSE_PORT", "8000")
SERVER_NAME = "redis"
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "sse")

connections = {
        SERVER_NAME: {
            "command": sys.executable,
            "args": [str(MCP_SCRIPT)],
            "env": {
                "REDIS_HOST": os.getenv("REDIS_HOST", "127.0.0.1"),
                "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
                "TRANSPORT": "sse",
            },
        }
    }

# ────────────────────────────────────────────────────────
# 5) build a Supervisor LangGraph agent
# ────────────────────────────────────────────────────────
# Bring all the agents togather - Supervisro Agent
#research_supervisor_node = make_supervisor_node(llm_oci, ["RAG", "Web_Scrapper", "search", "nl2sql", "nl2sql_sf"])

class MessageClassifier(BaseModel):
    message_type: Literal["rag", "sql"] = Field(
        ...,
        description="Classify if the message requires to be handled by a RAG agent or a SQL Agent",)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None


def extract_json(text: str) -> dict:
    """
    Extracts a valid JSON object from the text using regex.
    If multiple JSON objects exist, it picks the first one.
    """
    json_pattern = re.findall(r'\{.*?\}', text, re.DOTALL)

    for json_str in json_pattern:
        try:
            parsed_json = json.loads(json_str)

            # Handle nested JSON cases
            if isinstance(parsed_json, dict):
                if "supervisor" in parsed_json and "next" in parsed_json["supervisor"]:
                    return {"next": parsed_json["supervisor"]["next"]}
                if "next" in parsed_json:
                    return parsed_json
        except json.JSONDecodeError:
            continue  # Try the next match if this one fails

    return {"next": "FINISH"}  # Default if no valid JSON is found

def classify_message(state:State):
    last_message  = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)
    result = classifier_llm.invoke(
        [
            {"role":"system",
             "content": """ Classify the user message as either:
             - 'RAG' : if it ask for data from an unstructured system,
             - 'SQL' : if it ask for data from an structured system,
             """},
            {"role":"user", "content": last_message.content},
        ])

    return {"message_type": result.message_type}

def router(state:State):
    message_type = state.get("message_type", "sql")
    if(message_type == "sql"):
        return {"next":"rag"}

    return {"next": "sql"}

def rag_agent(state:State):
    last_message = state["messages"][-1]
    messages = [{"role":"system",
             "content": """ You are an RAG agent expert in handling unstructured data,
             """},
            {"role":"user", "content": last_message.content},]

    reply = llm.invoke(messages)
    return {"messages": [{"role":"assistant", "content": reply.content}]}

def sql_agent(state:State):
    last_message = state["messages"][-1]
    messages = [{"role": "system",
                 "content": """ You are an SQL agent expert in handling structured data,
                 """},
                {"role": "user", "content": last_message.content}, ]

    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


def chatbot(state: State):
    return {"messages" : llm.invoke(state["messages"])}

def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("classifier", classify_message)
    graph_builder.add_node("router", router)
    graph_builder.add_node("sql", sql_agent)
    graph_builder.add_node("rag", rag_agent)

    graph_builder.add_edge(START, "classifier")
    graph_builder.add_edge("classifier", "router")

    graph_builder.add_conditional_edges(
        "router",
        lambda state: state.get("next"),
        {"sql": "sql", "rag": "rag"},
    )
    graph_builder.add_edge("sql", END)
    graph_builder.add_edge("rag", END)

    graph = graph_builder.compile()

    return graph
def run_chatbot():
    state = {"messages": [], "message_type": None}
    graph = build_graph()

    while True:
        user_input = input("Messages: ")
        if user_input == "exit":
            print("Goodbye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if(state.get("messages" and len(state["messages"]) > 0)):
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


config = {"configurable": {"thread_id": "abc123"}}
async def build_agent():
    # configure the single Redis-MCP server
    async with MultiServerMCPClient(connections) as client:
        tools = client.get_tools()
        if not tools:
            raise RuntimeError(
                "No MCP tools found — make sure your server script is at "
                f"{MCP_SCRIPT} and that it calls mcp.run(transport='stdio'|'sse')."
            )


        llm_with_tools = llm.bind_tools(tools)

        SYSTEM_PROMPT = (
            "You are a Redis-savvy assistant. "
            "For reads: always use HGETALL.\n"
            "For writes: use HSET (and EXPIRE when needed)."
        )

        def supervisor(state: State):
            messages = state["messages"]

            # Insert system prompt only once
            if not any(isinstance(m, SystemMessage) for m in messages):
                messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

            return {"messages": [llm_with_tools.invoke(messages)]}


        # Build LangGraph
        builder = StateGraph(State)
        builder.add_node("supervisor", supervisor)
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("supervisor", tools_condition)
        builder.add_edge("tools", "supervisor")
        builder.add_edge(START, "supervisor")
        builder.add_edge("supervisor", END)

        graph = builder.compile(interrupt_before=[], interrupt_after=[])
        graph.name = "getinsight-supervisor"

        await getinsights(graph)
        return graph


# ────────────────────────────────────────────────────────
# 6) REPL that strips out any non-string AIMessage.content
# ────────────────────────────────────────────────────────
# ─── simple REPL ────────────────────────────────────
async def getinsights(agent, max_history: int = 30):
    history: deque[HumanMessage|AIMessage] = deque(maxlen=max_history)
    print("🔧  GetInsights Supervisor — type 'exit' to quit\n")
    while True:
        user_text = input("❓> ").strip()
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue

        history.append(HumanMessage(content=user_text))
        result = await agent.ainvoke({"messages": list(history)})
        ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        reply = ai_msg.content if ai_msg else "⚠️ (no reply)"
        print(f"\n🤖 {reply}\n")
        history.append(AIMessage(content=reply))

async def main():
    # 1) build your graph
    graph = await build_agent()
    # 2) enter the REPL
    #await getinsights(graph)

if __name__ == "__main__":
    run_chatbot()
    #asyncio.run(build_graph())



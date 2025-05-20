import oci
import os
# ─── LLM from OCI GenAI Services - Config  ──────────────────────────────
from langchain_community.chat_models import ChatOCIGenAI

 # ─── For creating prompts  ──────────────────────────────
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
from pathlib import Path

# ─── Langraph  ──────────────────────────────
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.prebuilt import create_react_agent

# ─── MCP  ──────────────────────────────
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

# ─── NVIDIA Nemo Guardrails imports ──────────────────────────────
# python -m pip install nemoguardrails
from nemoguardrails import LLMRails, RailsConfig

# ────────────────────────────────────────────────────────────────
# Load environment variables from .env file
# ────────────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env


# ────────────────────────────────────────────────────────────────
# Set your OCI credentials
# ────────────────────────────────────────────────────────────────

llm_oci = None


# ────────────────────────────────────────────────────────────────
# Define the state structure for our supervisor agent
# ────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ────────────────────────────────────────────────────────────────
# Set up LangSmith for LangGraph development
# ────────────────────────────────────────────────────────────────

from langsmith import Client
client = Client()
url = next(client.list_runs(project_name="anup-blog-post")).url
print(url)
print("LangSmith Tracing is Enabled")

# ────────────────────────────────────────────────────────────────
# Configure Nvidia Nemo Guardrails
# ────────────────────────────────────────────────────────────────
# TBD - https://github.com/nagarajjayakumar/cohere-nemo-demo/tree/main

def get_file_path(filename):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)

rails_config = RailsConfig.from_content(
        colang_content=open(get_file_path('nemo_guardrails/rails.config'), 'r').read(),
        yaml_content=open(get_file_path('nemo_guardrails/config.yml'), 'r').read()
    )

#rails = LLMRails(rails_config, llm_oci)
#response = await rails.generate_async(prompt=prompt_template)

# ────────────────────────────────────────────────────────────────
# Initialize the OCI LLM Service to be used for ReAct capabilities
# ────────────────────────────────────────────────────────────────
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
# Compile the LangGraph agent (needs an open MCP session)
# ────────────────────────────────────────────────────────────────
async def build_agent(session: ClientSession):
    tools = await load_mcp_tools(session)
    return create_react_agent(llm_oci, tools)
# TBD - Use MCP Tools and Not Function Call

# ────────────────────────────────────────────────────────────────
# Setting up the Graph and Tools
# ────────────────────────────────────────────────────────────────

def setup_graph():

    # Initialize our state graph
    graph_builder = StateGraph(State)

    # Set up the search tool
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # 3) Initialize a fresh LLM here
    llm = initialize_llm()

    # Connect the tools to our AI model
    llm_with_tools = llm.bind_tools(tools)

    # Define the supervisor node function
    def supervisor(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build the graph structure
    graph_builder.add_node("supervisor", supervisor)
    graph_builder.add_node("tools", ToolNode(tools=[tool]))
    graph_builder.add_conditional_edges("supervisor", tools_condition)
    graph_builder.add_edge("tools", "supervisor")
    graph_builder.add_edge(START, "supervisor")

    return graph_builder.compile()

# ────────────────────────────────────────────────────────────────
# Creating a Test Function
# ────────────────────────────────────────────────────────────────
def test_chatbot(graph, question: str):
    print("\n" + "="*50)
    print(f"😀 User: {question}")
    print("="*50)

    try:
        for event in graph.stream({"messages": [("human", question)]}):
            for value in event.values():
                if "messages" in value:
                    message = value["messages"][-1]
                    if hasattr(message, "content"):
                        print("\n🤖 AI:", message.content)
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        print("\n🔍 Searching...")
                        for tool_call in message.tool_calls:
                            print(f"- Search query: {tool_call['args'].get('query', '')}")
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")


# ────────────────────────────────────────────────────────────────
# The Main Execution Code
# ────────────────────────────────────────────────────────────────
def main():
    test_questions = [
        "Hello! Good Morning !",
        "What are three popular movies right now?",
    ]


    print("🔄 Initializing chatbot...")
    graph = setup_graph()
    print("✅ Chatbot ready!\n")

    for question in test_questions:
        test_chatbot(graph, question)
        print("\n" + "-" * 50)

# Run the main function
if __name__ == "__main__":
    main()
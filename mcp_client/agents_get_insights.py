import oci
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
venv_root = Path("/Users/aojah/PycharmProjects/mcp_redis/.venv/.env")   # set automatically on activation
load_dotenv(venv_root)


# ────────────────────────────────────────────────────────────────
# Set your OCI credentials
# ────────────────────────────────────────────────────────────────

llm_oci = None

COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaau6esoygdsqxfz6iv3u7ghvosfskyvd6kroucemvyr5wzzjcw6aaa"
AUTH_TYPE = "API_KEY" # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
CONFIG_PROFILE = "DEFAULT"

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
#model_id = "meta.llama-3.3-70b-instruct"
model_id = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq"

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
# Load a guardrails configuration from the specified path.
#config = RailsConfig.from_path("/Users/aojah/PycharmProjects/mcp_redis/nemo_guardrails/config.yml")
#rails = LLMRails(config, llm_oci)
#response = await rails.generate_async(prompt=prompt_template)

# ────────────────────────────────────────────────────────────────
# Initialize the OCI LLM Service to be used for ReAct capabilities
# ────────────────────────────────────────────────────────────────
def initialize_llm():
    try:
        llm_oci = ChatOCIGenAI(
            model_id=model_id,
            service_endpoint=endpoint,
            compartment_id=COMPARTMENT_ID,
            provider="cohere", # Create an OCI Cohere LLM instance
            model_kwargs={
                "temperature": 1,
                "max_tokens": 600,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "top_k": 0,
                "top_p": 0.75
            },
            auth_type=AUTH_TYPE,
            auth_profile=CONFIG_PROFILE
        )
        return llm_oci
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        raise

# ────────────────────────────────────────────────────────────────
# Compile the LangGraph agent (needs an open MCP session)
# ────────────────────────────────────────────────────────────────
async def build_agent(session: ClientSession):
    tools = await load_mcp_tools(session)
    return create_react_agent(llm_oci, tools)

# ────────────────────────────────────────────────────────────────
# Setting up the Graph and Tools
# ────────────────────────────────────────────────────────────────

def setup_graph():

    # Initialize our state graph
    graph_builder = StateGraph(State)

    # Set up the search tool
    tool = TavilySearchResults(max_results=2)
    tools = [tool]

    # Set up the AI model
    llm = llm_oci

    # Connect the tools to our AI model
    llm_with_tools = llm.bind_tools(tools)

    # Define the chatbot node function
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Build the graph structure
    graph_builder.add_node("supervisor", chatbot)
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
    # Set up the AI model
    llm_oci = initialize_llm()
    main()

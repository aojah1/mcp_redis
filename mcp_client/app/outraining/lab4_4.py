from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from langchain.prompts import PromptTemplate
import os
from pathlib import Path
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.utilities import WikipediaAPIWrapper
# python3.13 -m pip install openpyxl





# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent

############ Task 1

from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.react.base import ReActAgent
from llama_index.core.settings import Settings
from llama_index.llms.cohere import Cohere

#%% md
# Load Documents

docs_path = Path(f"{PROJECT_ROOT}/docs/")
documents = SimpleDirectoryReader(docs_path).load_data()

#%% md
# Create Index and Query Engine

from llama_index.embeddings.cohere import CohereEmbedding
COHERE_API_KEY="SnW0xkwMGxrH7IjwZ9lK9y5DYSmvvDRhUYtJ4jxG"
llm = Cohere(api_key=COHERE_API_KEY, model="command-r-plus-08-2024")

Settings.llm = llm  # Global LLM setting for all components
embed_model = CohereEmbedding(
    cohere_api_key=COHERE_API_KEY,
    model_name="embed-english-v3.0",
    input_type="search_document",
)


index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)

query_engine = index.as_query_engine()

#%% md
# Register Tool

qa_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="multi_doc_search_tool",
        description=(
            "Use this tool to search finance_policy and customer_support_guidelines "
            "for questions about policies, procedures, RAG, or customer service strategy."
        )
    )
)

#%% md
# Create Agent

agent = ReActAgent.from_tools([qa_tool], llm=llm, verbose=False)

print("==Agentic RAG with LlamaIndex==\n")

user_query = "What is the expected response time for emails and chats?"
print(f"Query: {user_query}\n")

response = agent.chat(user_query)

print("\nFinal Response:")
print(response.response)

#%% md
#### Add-on Function to display source documents


def print_source_documents(agent_response):
    if not hasattr(agent_response, "sources") or not agent_response.sources:
        print("⚠️ No source documents found in the response.")
        return

    print("📄 Source Documents:\n")
    for source_index, source in enumerate(agent_response.sources, start=1):
        # Step 1: Navigate into `raw_output.source_nodes`
        if hasattr(source, "raw_output") and hasattr(source.raw_output, "source_nodes"):
            for node_index, node_with_score in enumerate(source.raw_output.source_nodes, start=1):
                node = node_with_score.node
                score = node_with_score.score
                metadata = node.metadata

                print(f"--- Source Document {source_index}.{node_index} ---")
                print(f"🗂 File Name      : {metadata.get('file_name')}")
                print(f"📁 File Path      : {metadata.get('file_path')}")
                print(f"📝 File Type      : {metadata.get('file_type')}")
                print(f"📅 Created On     : {metadata.get('creation_date')}")
                print(f"🧮 File Size      : {metadata.get('file_size')} bytes")
                print(f"📊 Similarity Score: {score:.4f}")
                print(f"\n📃 Document Content Preview:\n{node.text.strip()[:500]}...\n")
                print("-" * 60)
        else:
            print(f"⚠️ Source {source_index} does not contain `raw_output.source_nodes`.")

print_source_documents(response)
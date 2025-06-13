from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from langchain.prompts import PromptTemplate
import os
from pathlib import Path

# python3.13 -m pip install openpyxl

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.document_loaders import TextLoader

llm = initialize_llm()
embedding_model = initialize_embedding_model()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


# ########## Task 1
#
# #%% md
# ### Load PDF
#
# loader = PyPDFLoader(f"{PROJECT_ROOT}/finance_data.pdf")
# pages = loader.load()
#
# #%% md
# ### Combined raw text from all pages
# raw_text = ''
#
# for i, doc in enumerate(pages):
#     text = doc.page_content
#     if text:
#         raw_text += text
#
# #%% md
# ### Text splitter
#
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100
# )
#
# texts = text_splitter.split_text(raw_text)
#
# #%% md
# ### Text chunk into a document structure
#
#
# from langchain_core.documents import Document
#
# docs = []
# for i in range(len(texts)):
#     doc = Document(page_content=texts[i])
#     docs.append(doc)
#
# #%% md
# ### Initialize a vector database
#
#
# vectordb = Chroma(
#     collection_name='summaries',
#     embedding_function=embedding_model,
#     persist_directory=f'{PROJECT_ROOT}/data'
# )
#
# vectordb.add_documents(docs)
#
# #%% md
# ### Create a retriever
#
# retriever = vectordb.as_retriever(search_kwargs={"k": 4})
#
# #%% md
# ### Define a prompt template
#
#
# PRODUCT_BOT_PROMPT = """
#     You are a smart assistant.
#     Your response must only be in English.
#     Ensure your answers are relevant to the query with reference to provided context and not outside the context.
#     Your responses should be elaborate and up to the mark referring to the context only.
#     Do not include the keyword context in the final answer
#
#     CONTEXT:
#     {context}
#
#     QUESTION: {question}
#
#     YOUR ANSWER:
# """
#
# from langchain_core.prompts import ChatPromptTemplate
#
# prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_PROMPT)
#
# #%% md
# ### Chaining inputs/outputs
#
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
#
#
# # Define the full processing pipeline
# # 1. Takes query input
# # 2. Retrieves relevant context from vector DB
# # 3. Fills in the prompt template
# # 4. Sends it to the LLM
# # 5. Parses the output string
#
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
#
# #%% md
# ### Process the query
#
# """ Example Questions to try
# Who is Mr. Raza
# Who is Steve Jobs
# What is Form 10 - K
# Explain acquisitions
# List different events from year 2020 to 2023
# """
#
# query="List different events from year 2020 to 2023"
#
# #result = chain.invoke(query)
#
# Print response
# print("Response:", result)

########## Task 2
# Agentic RAG Finance Demo

#%% md
## Load, Split, Embed, and Index Documents




# Read data files
docs_folder = f"{PROJECT_ROOT}/financial_docs"

loaders = [TextLoader(os.path.join(docs_folder, f), encoding="utf8")
           for f in os.listdir(docs_folder) if f.endswith(".txt")]

all_docs = []
for loader in loaders:
    all_docs.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(all_docs)

vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=f"{PROJECT_ROOT}/chromadb")
print(vectorstore)

## Build RetrievalQA and Define RAG Tool



qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    return_source_documents=True
)

def financial_qa(query: str) -> str:
    """Tool function: runs QA and appends source citations."""
    result = qa_chain({"query": query})
    answer = result["result"]
    sources = result["source_documents"]
    citation_lines = []
    for doc in sources:
        src = os.path.basename(doc.metadata.get("source", "unknown"))
        citation_lines.append(f"- {src} (page chunk)")
    citations = "\n".join(citation_lines)
    return f"{answer}\n\nSources:\n{citations}"

## Register Tool and Initialize Agent

# Define Summarization Tools

tools = [
    Tool(
        name="FinancialRAG",
        func=financial_qa,
        description="Use for answering questions about our Q1 2024 financials and market analysis, with source references."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True,
    verbose=True
)

# ## Pass a query to the Agent
# ### Query1
# query = "Summarize our net income performance in Q1 2024."
#
# response = agent.invoke(query)
# print("\n=== Agent Response ===\n")
#
# from IPython.display import Markdown, display
# print(response['input'],"\n")
# print(response['output'])
#
# #Query 2
# query = "What was our YoY revenue growth in Q1 2024?"
#
# response = agent.invoke(query)
# print("\n=== Agent Response ===\n")
#
# from IPython.display import Markdown, display
# print(response['input'],"\n")
# print(response['output'])
#
# # Query 3
# # Section 8
# query = "What was our YoY revenue growth in Q1 2024, and what market volatility trends were noted in the Market_Analysis_2024 report?"
#
# response = agent.invoke(query)
# print("\n=== Agent Response ===\n")
#
# from IPython.display import Markdown, display
# print(response['input'],"\n")
# print(response['output'])
#
# #Query 4
# query = "According to the Market_Analysis_2024 report, what were the key market trends observed in Q1 2024?"
#
# response = agent.invoke(query)
# print("\n=== Agent Response ===\n")
#
# from IPython.display import Markdown, display
# print(response['input'],"\n")
# print(response['output'])

########## Task 3
from langchain.chains import LLMChain
# Agentic RAG Finance Demo with Multiple Tools
#%% md
## Build a summarization chain and tool functions

summary_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "You are a finance-domain expert. "
        "Provide a concise summary of the following document:\n\n{text}"
    )
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

def list_documents(_: str) -> str:
    """Lists all source files in the docs folder."""
    files = os.listdir(docs_folder)
    return "Available documents:\n" + "\n".join(f"- {f}" for f in files)

def summarize_financial_report(_: str) -> str:
    """Summarizes the Q1 2024 financial report."""
    path = os.path.join(docs_folder, "Q1_2024_Financial_Report.txt")
    with open(path, encoding="utf8") as f:
        content = f.read()
    return summary_chain.run(text=content)

def summarize_market_analysis(_: str) -> str:
    """Summarizes the 2024 market analysis report."""
    path = os.path.join(docs_folder, "Market_Analysis_2024.txt")
    with open(path, encoding="utf8") as f:
        content = f.read()
    return summary_chain.run(text=content)

# ── Wrap everything in an agent ─────────────────────────────────────────────
tools = [
    Tool(
        name="FinancialRAG",
        func=financial_qa,
        description=(
            "Answer detailed finance questions about Q1 2024 "
            "and provide source citations."
        )
    ),
    Tool(
        name="ListDocuments",
        func=list_documents,
        description="List all available source documents."
    ),
    Tool(
        name="SummarizeFinancialReport",
        func=summarize_financial_report,
        description="Provide a concise summary of the Q1 2024 financial report.Used only when user explicitly requests summary on Finance report"
    ),
    Tool(
        name="SummarizeMarketAnalysis",
        func=summarize_market_analysis,
        description="Provide a concise summary of the 2024 market analysis report.Used only when user explicitly requests summary on market analysis report"
    ),
]


agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True,
    verbose=True
)

#%% md
## Pass a query to the Agent
### Query1

# # query = "Summarize our net income performance in Q1 2024."
# query = "SummarizeFinancialReport"
#
# response = agent.invoke(query)
# print("\n=== Agent Response ===\n")
#
# from IPython.display import Markdown, display
# print(response['input'],"\n")
# print(response['output'])

# Query 2

query = "ListDocuments and also SummarizeFinancialReport"

response = agent.invoke(query)
print("\n=== Agent Response ===\n")

from IPython.display import Markdown, display
print(response['input'],"\n")
print(response['output'])

############ Task 4

## Create a chain that combines all retrieved documents into one prompt feeds to context
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant for finance analysis."),
    ("human", "Use the following financial data:\n{context}\n\nNow answer:\n{question}")
])

stuff_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)


from langchain_core.runnables import RunnableSequence
from langchain_core.runnables import RunnableLambda, RunnableMap

#%% md
# Wrap it into a RAG pipeline


#-------------Option2 returning the source document------------
# Define the RAG pipeline logic: retrieve → generate → return
def rag_logic(input_dict):
    question = input_dict["question"]
    documents = retriever.get_relevant_documents(question)
    answer = stuff_chain.invoke({"context": documents, "question": question})
    return {
        "answer": answer,
        "sources": documents
    }

# Define the RAG pipeline logic: retrieve → generate → return

rag_chain = RunnableLambda(rag_logic)

#%% md
# Run the full RAG pipeline with a sample question

query = "Which companies had the highest revenue in 2023?"

response = rag_chain.invoke({"question": query})


rint("Answer:\n", response["answer"])

# Print the source documents used in the response

print("\n📂 Source Documents:")
for i, doc in enumerate(response["sources"], 1):
    print(f"\n--- ✅Document {i} ---\n{doc.page_content[:500]}...\n")

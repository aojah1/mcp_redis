import time
import re
from typing import TypedDict, Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from typing import TypedDict
import os

llm = initialize_llm()
embeddings = initialize_embedding_model()

# === Agent State Definition ===
class AgentState(TypedDict):
    input: str
    next: str
    output: str
    latency: float
    rag_result: str

# === Load Multiple Text Documents ===
file_paths = [
    "IT_Security_Knowledge_Base.txt",
    "Data_Protection_Guidelines.txt",
    "Remote_Work_Policy.txt"
]

# Load all documents
all_docs = []
for file in file_paths:
    loader = TextLoader(file)
    all_docs.extend(loader.load())

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

#embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key="uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# === Setup LLM + Memory ===
#llm = ChatCohere(model="command-r-plus", temperature=0.3, cohere_api_key="uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr")
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# === Intent Classifier ===
def classify_intent(state: AgentState) -> Dict[str, Any]:
    query = state["input"].lower()
    if any(term in query for term in [
        "policy", "mfa", "password", "backup", "vpn", "security",
        "encryption", "restore", "data", "remote", "access"
    ]):
        return {"next": "rag"}
    elif any(term in query for term in ["what did", "last", "memory", "remind", "previous"]):
        return {"next": "memory"}
    else:
        return {"next": "fallback"}

# === RAG Node ===
def rag_tool(state: AgentState) -> Dict[str, str]:
    docs = retriever.invoke(state["input"])
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""Use the following policy context to answer the user's question:

{context}

Q: {state['input']}
A:"""
    response = llm.invoke(prompt).content
    return {"output": response.strip()}

# === Memory Node ===
def memory_tool(state: AgentState) -> Dict[str, str]:
    hist = memory.load_memory_variables({})["history"]
    user_msgs = [m.content for m in hist if m.type == "human"]
    if not user_msgs:
        return {"output": "🧠 I have no memory of previous queries."}
    return {"output": f"🧠 You previously asked: {user_msgs[-1]}"}

# === Fallback Node ===
def fallback_tool(state: AgentState) -> Dict[str, str]:
    return {"output": "⚠️ I couldn't understand. Please ask about IT policy, MFA, encryption, or backups."}

# === LangGraph Definition ===
builder = StateGraph(AgentState)
builder.add_node("classify", RunnableLambda(classify_intent))
builder.add_node("rag", RunnableLambda(rag_tool))
builder.add_node("memory", RunnableLambda(memory_tool))
builder.add_node("fallback", RunnableLambda(fallback_tool))

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", lambda s: s["next"], {
    "rag": "rag",
    "memory": "memory",
    "fallback": "fallback"
})
builder.add_edge("rag", END)
builder.add_edge("memory", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# === Sample Test Queries ===
queries = [
    "What is the company policy on MFA?",
    "Are there any standards for creating passwords?",
    "Is customer data encrypted during transmission?",
    "How do we access internal tools remotely?",
    "Remind me what I asked about MFA.",
    "What’s the cafeteria policy?" # fallback
]

print("\n📊 RAG Agent Responses\n" + "-"*30)
for query in queries:
    start = time.time()
    result = graph.invoke({
        "input": query,
        "rag_result": "",
        "next": "",
        "output": "",
        "latency": 0.0
    })
    end = time.time()
    memory.save_context({"input": query}, {"output": result["output"]})
    print(f"\n💬 {query}")
    print(f"🤖 {result['output']}")
    print(f"⏱ Latency: {round(end - start, 3)} sec")



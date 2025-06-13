
from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from langchain.prompts import PromptTemplate
import os
from pathlib import Path

# python3.13 -m pip install openpyxl

from langchain_community.document_loaders import ImageCaptionLoader
#!pip install transformers langchain-chroma
#!pip install transformers torch==2.6
#!pip install faiss-cpu

llm = initialize_llm()
embedding_model = initialize_embedding_model()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


########## Task 1

#%% md
### 📚 Concept: Vector Databases and Similarity Search
# Import necessary libraries
from langchain_community.document_loaders import TextLoader
# Import necessary libraries
from langchain_community.vectorstores import FAISS
# Import necessary libraries

# Import necessary libraries
from langchain_text_splitters import CharacterTextSplitter

# Load the text file using UTF-8 encoding
text_loader = TextLoader(f"{PROJECT_ROOT}/speech.txt", encoding="utf-8")
raw_documents = text_loader.load()

#%% md
### Split the document into smaller chunks for embedding
# Split the document into manageable chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
chunked_documents = splitter.split_documents(raw_documents)

#%% md
### Create a FAISS vector store from the documents and embeddings
vector_store = FAISS.from_documents(chunked_documents, embedding_model)

#%% md
### Query definition
# Define a natural language query
search_query = "How does the speaker describe the desired outcome of the war?"

#%% md
### 1. Simple similarity search (returns list of relevant documents)

# Perform a similarity search to find most relevant documents
similar_documents = vector_store.similarity_search(search_query)
# Display the result with content preview
print("\n🔍 Most Relevant Document Content:\n", similar_documents[0].page_content)

#%% md
### 2. Using the retriever interface for querying

retriever = vector_store.as_retriever()
retrieved_documents = retriever.invoke(search_query)

# Display the result with similarity score and content preview
print("\n📄 Retrieved Document:\n", retrieved_documents[0].page_content)

#%% md
### 3. Similarity search with similarity scores
documents_with_scores = vector_store.similarity_search_with_score(search_query)

# Display the result with similarity score and content preview
print("\n📊 Similarity Search with Scores:")

for idx, (doc, score) in enumerate(documents_with_scores, start=1):
# Display the result with similarity score and content preview
    print(f"\n--- Result {idx} ---")
# Display the result with similarity score and content preview
    print(f"🔢 Similarity Score: {score:.4f}")
# Display the result with similarity score and content preview
    print(f"📄 Document Excerpt:\n{doc.page_content[:500]}...\n")

#%% md
### 4. Convert the query to embedding and search using the vector
# Define a natural language query
search_query = "How does the speaker describe the desired outcome of the war?"
# Convert the query into a vector embedding
query_vector = embedding_model.embed_query(search_query)

vector_based_results = vector_store.similarity_search_by_vector(query_vector)
# Display the result with similarity score and content preview
print("\n📈 Vector-Based Search Results:\n", vector_based_results)

#%% md
### 5. Persist the FAISS index to disk (optional)
vector_store.save_local("faiss_index")

#%% md
### 6. Load the persisted index for future use

loaded_vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

#%% md
### 7. Perform a new search on the loaded index
# Perform a similarity search to find most relevant documents
final_results = loaded_vector_store.similarity_search(search_query)
# Display the result with similarity score and content preview
print("\n📁 Final Search Results:\n", final_results[0].page_content)

#%% md
## FAISS index for cosine similarity

# Import necessary libraries
from langchain_community.document_loaders import TextLoader
# Import necessary libraries
from langchain_community.vectorstores import FAISS
# Import necessary libraries

# Import necessary libraries
from langchain_text_splitters import CharacterTextSplitter
# Import necessary libraries
import faiss
# Import necessary libraries
import numpy as np

# Step 1: Load the text document
# Load the text file using UTF-8 encoding
text_loader = TextLoader(f"{PROJECT_ROOT}/speech.txt", encoding="utf-8")
raw_docs = text_loader.load()

# Step 2: Split the document into chunks
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
# Split the document into smaller chunks for embedding
documents = splitter.split_documents(raw_docs)


# Step 4: Get document embeddings and normalize them
texts = [doc.page_content for doc in documents]
# Convert document text to embeddings
embeddings = np.array(embedding_model.embed_documents(texts)).astype("float32")
faiss.normalize_L2(embeddings)

# Step 5: Create FAISS index for cosine similarity
# Create a FAISS index using inner product (for cosine similarity)
index = faiss.IndexFlatIP(embeddings.shape[1])
# Add embeddings to the FAISS index
index.add(embeddings)


# Step 6: Convert query to normalized 2D embedding
# Define a natural language query
query = "How does the speaker describe the desired outcome of the war?"
# Convert the query into a vector embedding
query_vector = np.array(embedding_model.embed_query(query)).astype("float32").reshape(1, -1)
faiss.normalize_L2(query_vector)

# Step 7: Search for similar documents (top 3 results)
# Perform a similarity search to find most relevant documents
scores, indices = index.search(query_vector, k=3)

# Step 8: Show results
# Display the result with similarity score and content preview
print("\n📊 Cosine Similarity Search Results:")
for rank, idx in enumerate(indices[0], start=1):
    doc = documents[idx]
    score = scores[0][rank - 1]
# Display the result with similarity score and content preview
    print(f"\n--- Result {rank} ---")
# Display the result with similarity score and content preview
    print(f"🧭 Cosine Similarity Score: {score:.4f}")
# Display the result with similarity score and content preview
    print(f"📄 Document Preview:\n{doc.page_content[:500]}...\n")
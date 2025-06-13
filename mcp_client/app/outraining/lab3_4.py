from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from llm.oci_embedding_model import initialize_embedding_model
from langchain.prompts import PromptTemplate
import os
from pathlib import Path

# python3.13 -m pip install openpyxl


llm = initialize_llm()
embedding_model = initialize_embedding_model()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


################ Task 1

# Understanding Embeddings

# Sample input documents to be embedded
greek_alphabet_sentences = [
    "Alpha is the first letter of Greek alphabet",
    "Beta is the second letter of Greek alphabet",
]

# Generate embeddings for the input documents
document_embeddings = embedding_model.embed_documents(greek_alphabet_sentences)

# Get embedding vector for the first document
first_embedding_vector = document_embeddings[0]

# Display the length of the first embedding vector
print("Length of the first embedding vector:", len(first_embedding_vector))
print()

# Display the embedding vector for the first document

print("First embedding vector:", first_embedding_vector[:50])

################# Task 2

##%% md

# 📄 LangChain Document Loaders Lab
# !pip install -qU wikipedia arxiv
# !pip install -qU pymupdf, pypdf
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader, ArxivLoader, WikipediaLoader

#%% md
### Load a plain text file related to healthcare domain
#
# Converts the file content into LangChain Document objects
# Load from a plain text file (Healthcare Domain)
text_loader = TextLoader(f"{PROJECT_ROOT}/healthcare_policy.txt")
healthcare_docs = text_loader.load()
print(f"\n📘 Healthcare Doc Sample:\n\n{healthcare_docs[0].page_content[:300]}")

#%% md
### Load a PDF file related to insurance guidelines

#
# Parses PDF pages into LangChain Document objects
# Load from a PDF file (Insurance Domain)
pdf_loader = PyPDFLoader(f"{PROJECT_ROOT}/insurance_guidelines.pdf")
insurance_docs = pdf_loader.load()
print(f"\n📄 Insurance Doc Sample:\n\n{insurance_docs[0].page_content[:300]}")

# #%% md
# ### Load data from a publicly available e-commerce related webpage
# Useful for ingesting blog articles or domain-specific documentation
# Load content from a web page (E-Commerce Domain)
web_loader = WebBaseLoader("https://www.shopify.com/blog/what-is-ecommerce")
ecommerce_docs = web_loader.load()
print(f"\n🛒 E-Commerce Web Doc Sample:\n\n{ecommerce_docs[0].page_content[:300]}")

# #%% md
# ### Load a research paper from arXiv (machine learning domain)
# Returns abstract and content from top matched result
# Load from Arxiv (Scientific Research – e.g., Machine Learning)
arxiv_loader = ArxivLoader(query="machine learning", load_max_docs=1)
arxiv_docs = arxiv_loader.load()
print(f"\n📚 Arxiv Paper Sample:\n{arxiv_docs[0].page_content[:300]}")

# #%% md
# ### Load content from Wikipedia (e.g., Health Insurance article)

# Helpful for summarization or knowledge extraction tasks
# Load from Wikipedia (Finance Domain)
wiki_loader = WikipediaLoader(query="Health insurance", lang="en",load_max_docs=4)
wiki_docs = wiki_loader.load()
print(f"\n🌐 Wikipedia Article Sample:\n{wiki_docs[0].page_content[:300]}")


############### Task 3

#%% md
#  LangChain Text Splitters – Concepts and Explanation
# pip install langchain langchain-text-splitters beautifulsoup4

from langchain_text_splitters import (
# Split text using newline with fixed chunk size and overlap
    RecursiveCharacterTextSplitter,
# Split text using newline with fixed chunk size and overlap
    CharacterTextSplitter,
# Split HTML content by <h1> tags
    HTMLHeaderTextSplitter,
# Flatten JSON object hierarchies into text chunks
    RecursiveJsonSplitter
)

from pathlib import Path
import json
import requests


base_dir = Path(PROJECT_ROOT)  # Replace with your path
text_path = base_dir / "wild_horses_story.txt"
html_path = base_dir / "historical_events.html"

#text_path = f"{PROJECT_ROOT}/wild_horses_story.txt"
#html_path = f"{PROJECT_ROOT}/historical_events.html"

text_content = text_path.read_text()
html_content = html_path.read_text()
json_content=requests.get("https://api.smith.langchain.com/openapi.json").json()


#%% md
### 1. CharacterTextSplitter

# Split text using newline with fixed chunk size and overlap
char_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
char_chunks = char_splitter.split_text(text_content)

print("=== CharacterTextSplitter ===")
for i, chunk in enumerate(char_chunks[:2], 1):
# Display chunk samples
    print(f"\nChunk {i}:\n{chunk[:300]}...\n")

#%% md
### 2. RecursiveCharacterTextSplitter

# Split text using newline with fixed chunk size and overlap
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
recursive_chunks = recursive_splitter.split_text(text_content)

print("=== RecursiveCharacterTextSplitter ===")
for i, chunk in enumerate(recursive_chunks[:2], 1):
# Display chunk samples
    print(f"\nChunk {i}:\n{chunk[:300]}...\n")

#%% md
### 3. HTMLHeaderTextSplitter

# Split HTML content by <h1> tags
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "section")])
html_chunks = html_splitter.split_text(html_content)


print("=== HTMLHeaderTextSplitter ===")
for i, chunk in enumerate(html_chunks[:2], 1):
# Display chunk samples
    print(f"\nChunk {i}:\n{chunk.page_content[:300]}...\n")

#%% md
### 4. RecursiveJsonSplitter

# Flatten JSON object hierarchies into text chunks
json_splitter = RecursiveJsonSplitter(max_chunk_size=300)
# json_chunks = json_splitter.split_json(json_content)
json_chunks=json_splitter.create_documents(texts=[json_content])


print("=== RecursiveJsonSplitter ===")
for i, chunk in enumerate(json_chunks[:3],1):
# Display chunk samples
    print(f"\nChunk {i}:\n{chunk.page_content[:300]}...\n")

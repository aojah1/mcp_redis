
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


llm = initialize_llm()
embedding_model = initialize_embedding_model()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent


#%% md
### Image Analysis

from langchain_community.document_loaders import ImageCaptionLoader

#  load images from data folder
list_image_urls = [
    f"{PROJECT_ROOT}/birdimg.jpg",f"{PROJECT_ROOT}/carimg.jpg"
    ]

#  getting captions for images using ImageCaptionLoader() from LangChain

loader = ImageCaptionLoader(images=list_image_urls,)

list_docs = loader.load()
print(list_docs)

#%% md
## Display the image

import requests
from PIL import Image

Image.open(list_image_urls[0]).convert("RGB")

import requests
from PIL import Image

Image.open(list_image_urls[1]).convert("RGB")

#%% md
# loading image captions into vector db
from langchain_chroma import Chroma
#from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(list_docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#%% md
### Prompt design and interaction with LLM for image analysis

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#from langchain_cohere import ChatCohere
import warnings
warnings.filterwarnings('ignore')

model = llm
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#%% md
### Run the Query

response = rag_chain.invoke({"input": "What animals are in the images?"})

print(response["answer"])

response = rag_chain.invoke({"input": "What kind of images are there?"})

print(response["answer"])


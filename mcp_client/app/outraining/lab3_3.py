from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper
#pip install wikipedia
from llm.oci_genai import  initialize_llm
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import random

llm = initialize_llm()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent

################## Task 1

import requests
import wikipedia
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain.llms import Cohere

# === Tool: Wikipedia Summary Tool ===
@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a topic and return the summary."""
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Error: {str(e)}"


# === Tool: GitHub Public Repo Search ===
@tool
def github_repo_search(query: str) -> str:
    """Search GitHub repositories matching a keyword."""
    url = f"https://api.github.com/search/repositories?q={query}"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)
    repos = response.json().get("items", [])
    if not repos:
        return "No repositories found."
    top = repos[0]
    return f"Repo: {top['full_name']}, Stars: {top['stargazers_count']}, Description: {top['description']}"


# # === Tool: Open Library Book Search ===
@tool
def openlibrary_search(title: str) -> str:
    """Search books in OpenLibrary by title."""
    url = f"http://openlibrary.org/search.json?title={title}"
    response = requests.get(url)
    books = response.json().get("docs", [])
    if not books:
        return "No books found."
    book = books[0]
    return f"Title: {book.get('title')}, Author: {book.get('author_name', ['Unknown'])[0]}, First Published: {book.get('first_publish_year')}"

# === Register Selected Tools with Descriptions ===
tools = [
    Tool(
        name="Wikipedia Search",
        func=wikipedia_search,
        description="Useful for answering general knowledge or topic-related questions by summarizing Wikipedia articles."
    ),
    Tool(
        name="GitHub Repo Search",
        func=github_repo_search,
        description="Useful for finding public GitHub repositories based on a topic, including repo name, star count, and description."
    ),
    Tool(
        name="OpenLibrary Book Search",
        func=openlibrary_search,
        description="Useful for retrieving information about books, such as title, author, and publication year, using the book's title."
    ),
]

# === Create Agent ===
agent = initialize_agent(
    tools=tools,
    llm=llm,
     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_interactions=1,
)

# === Example Queries ===

"""

"What is quantum computing?"
"Find a GitHub repo about data visualization"
"Search for the book 'Pride and Prejudice'"
"Search for open-source health tracking tools and return the links to access them"

"""

query = "What is quantum computing??"

response = agent.run(query)
print(f"\n🤖 Agent Answer:\n{response}")

query = "Search for open-source health tracking tools and return the links to access them?"

response = agent.run(query)
print(f"\n🤖 Agent Answer:\n{response}")

query = "Search for the book 'Pride and Prejudice'"

response = agent.run(query)
print(f"\n🤖 Agent Answer:\n{response}")

query = "Find a GitHub repo about data visualization"

response = agent.run(query)
print(f"\n🤖 Agent Answer:\n{response}")
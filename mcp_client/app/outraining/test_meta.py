from langchain.agents import Tool
from langchain.utilities import WikipediaAPIWrapper
from llm.oci_genai import  initialize_llm
from langgraph.prebuilt import create_react_agent



# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────

llm = initialize_llm()

from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent

################## Task 1

#%% md
#### LangChain Agent Example: Use Calculator and Wikipedia Search
#### Task:  Answer a mixed query using both a calculator and Wikipedia.
"""
**The ReAct-based agent reads your query.**

**It chooses the right tool (calculator or Wikipedia).**

**Uses LLM to reason over tool outputs.**

**Returns a final synthesized answer.**
"""

# Define a custom calculator tool function
# It evaluates simple math expressions using Python's eval()
def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))  # Evaluate the expression and return as string
    except Exception as e:
        return f"Error in calculation: {e}"  # Catch and return any error as a string


# Create a Wikipedia tool using LangChain's WikipediaAPIWrapper
# This tool allows the agent to fetch factual information from Wikipedia
wiki = WikipediaAPIWrapper()


# Define the list of tools the agent can use
# Each tool is wrapped with the Tool class and includes:
# - name: for the agent to reference
# - func: the function to execute
# - description: tells the agent when it should use this tool

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving basic math expressions like '12*4' or 'sqrt(64)'."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for answering factual questions using Wikipedia content."
    )
]


# Initialize an agent using:
# - The defined tools
# - The Meta LLM
agent = create_react_agent(
            model=llm,
            tools=tools,
            name="expert",
            prompt="Answer a mixed query using both a calculator and Wikipedia.",
        )

# Define a mixed query that requires both math and factual knowledge
query = {"user": "What is 17 * 24 and who invented the telescope?"}

# Use .invoke() to send the query to the agent
response = agent.invoke(query)
print(response)
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper
#pip install wikipedia
from llm.oci_genai import  initialize_llm
from langchain.memory import ConversationBufferMemory

import os
from pathlib import Path

# python3.13 -m pip install openpyxl


llm = initialize_llm()

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
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
# - The Cohere LLM
# - A ReAct-style agent type that allows the LLM to decide what tool to call step-by-step
# - verbose=True to print internal thought process

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# Define a mixed query that requires both math and factual knowledge
query = "What is 17 * 24 and who invented the telescope?"

# Use .invoke() to send the query to the agent
response = agent.invoke(query)

# Print the final combined answer returned by the agent
# print("\nFinal Answer:", response)

print(response['input'],"\n")
print(response['output'])


################## Task 2

# #%% md
# ### Agent accessing Multiple user defined Tools ,based on user query

from datetime import datetime
import random

# 2. Tool: TimeNow
def get_time(_: str) -> str:
    return datetime.now().strftime("Current date and time is %A, %B %d, %Y at %H:%M:%S")

# 3. Tool: Quote of the Day
def inspirational_quote(_: str) -> str:
    quotes = [
        "The best way to predict the future is to invent it. – Alan Kay",
        "Code is like humor. When you have to explain it, it’s bad. – Cory House",
        "First, solve the problem. Then, write the code. – John Johnson",
        "Programs must be written for people to read, and only incidentally for machines to execute. – Harold Abelson"
    ]
    return random.choice(quotes)

# 4. Tool: Word Counter
def word_counter(text: str) -> str:
    count = len(text.strip().split())
    return f"The sentence contains {count} word{'s' if count != 1 else ''}."

# 5. Tool: Calculator
def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Calculation error: {e}"

# 6. Define tools with descriptions to guide the LLM
tools = [
    Tool(
        name="TimeNow",
        func=get_time,
        description="Use this tool to get the current date and time."
    ),
    Tool(
        name="QuoteOfTheDay",
        func=inspirational_quote,
        description="Use this tool when the user asks for a motivational or inspirational quote."
    ),
    Tool(
        name="WordCounter",
        func=word_counter,
        description="Use this tool to count the number of words in a given sentence."
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Use this tool to perform basic math calculations like '23 * 7 + 5'."
    )
]

# 7. Create LangChain agent to select appropriate tool based on user input
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True
)

# 8. Ask user for a query
print("=" * 60)
# query = input("🤖 Ask me anything (math, time, quote, or word count):\n> ")
# query = "Give the results for mathematical expression  (23 * 7 + 5)"
# query = "What time is it now?"
# query = "How many words are in: LangChain is awesome and helpful"

query = "Tell me something motivational and tell me what is the Time now"

print("Query:: ",query)
print("=" * 60)

# 9. Run the agent
response = agent.invoke(query)

# 10. Show the response

print(response['input'],"\n")
print(response['output'])

# #%% md
# ### Conversational Agent with Math + Wikipedia tools + Memory

# 2. Define a basic calculator function
def calculator_tool(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error in calculation: {e}"

# 3. Wikipedia search tool using LangChain utility
wiki = WikipediaAPIWrapper()

# 4. Create tool list
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for solving math expressions like '45*12' or 'sqrt(144)'."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for answering factual or historical questions using Wikipedia."
    )
]

# 5. Add conversational memory (stores history)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 6. Initialize the conversational agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,  # <== Conversation-friendly ReAct agent
    memory=memory,
    verbose=True
)

# 7. Simulate a multi-turn interaction
print("=" * 60)
print("🧠 Multi-turn Conversational Agent")
print("=" * 60)

# == Turn 1
q1 = "Who discovered penicillin?"
print("=" * 60)
print(q1)
print("=" * 60)

r1 = agent.invoke({"input": q1})
print("\n🤖 Agent Response:")
print("**"+r1['output']+"**")
print("\n",r1)

# == Turn 2: Follow-up referencing previous answer
q2 = "When did that happen?"
print("=" * 60)
print(q2)
print("=" * 60)

r2 = agent.invoke({"input": q2})
print("\n🤖 Agent Response:")
print("**"+r2['output']+"**")
print("\n",r2)


# == Turn 3: Math + factual combo
q3 = "Also, what is 85 multiplied by 4?"
print("=" * 60)
print(q3)
print("=" * 60)

r3 = agent.invoke({"input": q3})
print("\n🤖 Agent Response:")
print("**"+r3['output']+"**")

print("\n",r3)

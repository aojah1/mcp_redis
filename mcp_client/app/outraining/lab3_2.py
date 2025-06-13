from langchain.agents import initialize_agent, Tool, AgentType
from langchain.utilities import WikipediaAPIWrapper
#pip install wikipedia
from llm.oci_genai import  initialize_llm
from langchain.memory import ConversationBufferMemory
from datetime import datetime
import random
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

#  Define sample tools
def get_time(_: str) -> str:
    return datetime.now().strftime("It's %A, %d %B %Y, %I:%M:%S %p")

def inspirational_quote(_: str) -> str:
    quotes = [
        "Stay hungry, stay foolish. – Steve Jobs",
        "Talk is cheap. Show me the code. – Linus Torvalds"
    ]
    return random.choice(quotes)

tools = [
    Tool(name="GetTime", func=get_time, description="Get the current date and time."),
    Tool(name="GetQuote", func=inspirational_quote, description="Returns an inspirational quote.")
]

# query = "What is the current time and also tell me a quote?"
# ZERO_SHOT_REACT_DESCRIPTION
#  Best for: Simple tool reasoning without chat history.
agent1 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

query = "What is the current time and also tell me a quote?"

response1 = agent1.invoke(query)
print("ZERO_SHOT_REACT_DESCRIPTION Output:\n", response1)

#%% md
### 2. AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# Best for: Needing intermediate reasoning steps or structured output.
agent2 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
response2 = agent2.invoke(query)
print("STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION Output:\n", response2)

#%% md
## 3. AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION
# Best for: Multi-turn conversations that require memory context.
query = "What is the current time and also tell me a quote?"



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent3 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
response3 = agent3.invoke(query)
print("CHAT_CONVERSATIONAL_REACT_DESCRIPTION Output:\n", response3)


agent4 = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
response4 = agent4.invoke(query)
print("CONVERSATIONAL_REACT_DESCRIPTION Output:\n", response4)
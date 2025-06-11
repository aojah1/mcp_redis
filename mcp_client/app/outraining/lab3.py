from common.prompts import *
from debugpy.launcher.debuggee import describe
from llm.oci_genai import initialize_llm
from langchain.prompts import PromptTemplate
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

################# Task 1

from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
import warnings
from IPython.display import Markdown, display


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_AIresponse(chain, query: str) -> str:
    """Run the conversation chain on a user query."""
    return chain.invoke(query)

# ─────────────────────────────────────────────────────────────────────────────
# Memory‐backed Conversation Chain
# ─────────────────────────────────────────────────────────────────────────────
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

# ─────────────────────────────────────────────────────────────────────────────
#  Insurance Dialog
# ─────────────────────────────────────────────────────────────────────────────
# 1) Customer greets the system
# print("--- Customer → AI ---")
print(( "**--- Customer → AI ---**"))

print(( "**Hello, I’d like to check on my claim status**"))
# Seed the conversation
_ = conversation.run("Hello, I’d like to check on my claim status.")

# 2) Customer provides a claim number
query = "My claim number is CLM-20250610-1234. What’s the current status?"

print(( "**--- Customer → AI ---**"))
print(( "**"+query+"**"))

output = get_AIresponse(conversation, query)

print(( "**--- AI → Customer ---**"))

print(output['response'])

# 3) Customer asks what documents are still required
query = "Great, thanks. Which documents do I still need to upload to complete this claim?"

print(( "**--- Customer → AI ---**"))

print(( "**"+query+"**"))
output = get_AIresponse(conversation, query)

print(( "**--- AI → Customer ---**"))
print(output['response'])

# 4) Customer requests a summary of their policy coverages
query = (
    "Also, can you remind me what my policy covers? "
    "Specifically for roadside assistance and rental car coverage."
)

print(( "**--- Customer → AI ---**"))
print(( "**"+query+"**"))

output = get_AIresponse(conversation, query)

print(( "**--- AI → Customer ---**"))
print(output['response'])

# ─────────────────────────────────────────────────────────────────────────────
# 📋  Summarize Conversation Before Ending Chat
# ─────────────────────────────────────────────────────────────────────────────
# Build a summarization chain that ingests the full chat history
from langchain.prompts import PromptTemplate
summary_prompt = PromptTemplate.from_template(
    "Below is the full conversation between the customer and an insurance support AI:\n\n"
    "{collected_chat_history}\n\n"
    "Please summarize the customer's requests and key details so far in 3–4 bullet points."
)
# summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

summary_chain = summary_prompt | llm


# Pull the raw chat buffer from memory
chat_history = conversation.memory.buffer

# Generate and print the summary
summary = summary_chain.invoke({"collected_chat_history":chat_history})

print(( "**--- Conversation Summary (for internal use) ---**"))

print((summary.content))

print(conversation.memory.buffer)   # pint the content in memory i.e. chat history
conversation.memory.clear()    # clear the history if needed


################### Task 3

def get_AIresponse(chain, query):
    result = chain.run(query)
    return result

conversation = ConversationChain(
    llm=llm,
)

print(conversation.prompt.template)

conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation_buf("Good morning AI!")

response = get_AIresponse(
    conversation_buf,
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge.Get me only specific 3 details"
)

print(response)

response = get_AIresponse(
    conversation_buf,
    "I just want to analyze the different 3 possibilities. What can you think of?"
)
print(response)

response = get_AIresponse(
    conversation_buf,
    "Which data source types could be used to give context to the model?.Get me only 3 types"
)
print(response)

#%%
response = get_AIresponse(
    conversation_buf,
    "What is my aim again?"
)

print(response)

print(conversation_buf.memory.buffer)

conversation_buf.memory.clear()

print(conversation_buf.memory.buffer)

#%% md
### Memory type #2: ConversationSummaryMemory
#

conversation_sum = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm)
)

print(conversation_sum.memory.prompt.template)

response = get_AIresponse(
    conversation_sum,
    "Good morning AI!"
)
print(response)

response = get_AIresponse(
    conversation_sum,
     "My interest here is to explore the potential of integrating Large Language Models with external knowledge.Get me only specific 3 details"
)
print(response)

response = get_AIresponse(
    conversation_sum,
     "I just want to analyze the different 3 possibilities. What can you think of?"
)
print(response)

response = get_AIresponse(
    conversation_sum,
    "Which data source types could be used to give context to the model?.Get me only 3 types"
)
print(response)

response = get_AIresponse(
    conversation_sum,
    "What is my aim again?"
)
print(response)

#%% md
#**Lets See whats in memory**
#

print(conversation_sum.memory.buffer)

conversation_buf.memory.clear()
print(conversation_buf.memory.buffer)


#%% md
### Memory type #3: ConversationBufferWindowMemory
#

#%%
conversation_bufw = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1)
)

response = get_AIresponse(
    conversation_bufw,
    "Good morning AI!"
)
print(response)

response = get_AIresponse(
    conversation_bufw,
    "My interest here is to explore the potential of integrating Large Language Models with external knowledge.Get me only specific 3 details"
)
print(response)

response = get_AIresponse(
    conversation_bufw,
    "I just want to analyze the different 3 possibilities. What can you think of?"
)
print(response)

response = get_AIresponse(
    conversation_bufw,
    "Which data source types could be used to give context to the model?.Get me only 3 types"
)
print(response)

response = get_AIresponse(
    conversation_bufw,
    "What is my aim again?"
)
print(response)

#%% md
print("Observe it effectively `forgot` what we talked about in the first interaction. Let's see what it 'remembers'. Given that we set `k` to be `1`, we would expect it remembers only the last interaction.")
bufw_history = conversation_bufw.memory.load_memory_variables(
    inputs=[]
)['history']

print(bufw_history)



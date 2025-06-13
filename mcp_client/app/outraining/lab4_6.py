import uuid
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda


"""
Example Inputs to pass during interaction

Describe the issue	"The screen is flickering"
Affected product	"Smart LED TV"
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
#from langchain_cohere import ChatCohere
from llm.oci_genai import initialize_llm
from typing import TypedDict
import os

# Setup Cohere

# api_key_prod= "SnW0xkwMGxrH7IjwZ9lK9y5DYSmvvDRhUYtJ4jxG"
# llm = ChatCohere(model="command-r", temperature=0.4,cohere_api_key=api_key_prod)

llm = initialize_llm()

# Define state schema
class SupportState(TypedDict, total=False):
    step: str
    issue: str
    product: str
    resolution: str

# Node: Greet user
def greet(state):
    res = llm.invoke("You are a helpful customer support agent. Start the conversation.")
    print(f"\n🤖 {res.content}")
    return {"step": "ask_issue"}

# Node: Ask issue
def ask_issue(state):
    res = llm.invoke("Ask the customer what issue they are facing.")
    print(f"\n💬 {res.content}")
    issue = input("User: ")
    return {"issue": issue, "step": "ask_product"}

# Node: Ask product
def ask_product(state):
    res = llm.invoke("Ask the customer which product is affected.")
    print(f"\n📦 {res.content}")
    product = input("User: ")
    return {"product": product, "step": "resolve"}

# Node: Generate resolution
def resolve(state):
    prompt =(f"Suggest a resolution for issue '{state['issue']}' with product '{state['product']}' in 2-3 sentences."
              f"Generate a helpful response that includes the words 'battery', 'charging', and 'headphones' if they apply.")
    res = llm.invoke(prompt)
    print(f"\n✅ {res.content}")
    return {"resolution": res.content}

# Define graph
graph = StateGraph(SupportState)
graph.add_node("greet", RunnableLambda(greet))
graph.add_node("ask_issue", RunnableLambda(ask_issue))
graph.add_node("ask_product", RunnableLambda(ask_product))
graph.add_node("resolve", RunnableLambda(resolve))

graph.set_entry_point("greet")
graph.add_edge("greet", "ask_issue")
graph.add_edge("ask_issue", "ask_product")
graph.add_edge("ask_product", "resolve")
graph.add_edge("resolve", END)

agent = graph.compile()

from IPython.display import Image, display
display(Image(agent.get_graph().draw_mermaid_png()))

if __name__ == "__main__":
    print("📞 Customer Support Agent Ready")
    agent.invoke({})

import time
import re
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.runnables import RunnableLambda
#from langchain_cohere import ChatCohere
from llm.oci_genai import initialize_llm
from typing import TypedDict
import os

llm = initialize_llm()

def classify_intent(state):
    query = state["input"].lower()
    if "calculate" in query or "math" in query:
        return {"next": "math"}
    elif "search" in query:
        return {"next": "search"}
    else:
        return {"next": "fallback"}



def search_tool(state):
    return {"output": f" Simulated search results for: {state['input']}"}


def math_tool(state):
    return {"output": f" Simulated math result for: {state['input']}"}


def fallback_response(state):
    return {"output": f" Sorry, I couldn't classify your request."}


from typing import TypedDict

class AgentState(TypedDict):
    input: str
    next: str
    output: str

builder = StateGraph(AgentState)


builder.add_node("classify", RunnableLambda(classify_intent))
builder.add_node("search", RunnableLambda(search_tool))
builder.add_node("math", RunnableLambda(math_tool))
builder.add_node("fallback", RunnableLambda(fallback_response))

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", lambda s: s["next"], {
    "search": "search",
    "math": "math",
    "fallback": "fallback"
})

builder.add_edge("search", END)
builder.add_edge("math", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# from IPython.display import Image, display
# display(Image(graph.get_graph().draw_mermaid_png()))


test_cases = [
    {"input": "Please calculate the square root of 144", "next": "math"},
    {"input": "Search for recent AI trends", "next": "search"},
    {"input": "What's up?", "next": "fallback"}
]

for case in test_cases:
    result = graph.invoke(case)
    print(f" Input: {case['input']}")
    print(f" Output: {result['output']}\n")
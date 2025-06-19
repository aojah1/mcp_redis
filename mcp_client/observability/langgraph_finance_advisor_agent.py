# Purpose: Helps user with financial planning and investment suggestions
# langgraph_finance_advisor_agent.py
"""
Goal: "Plan for retirement in 20 years"
Risk: "high"

or

Goal: "Build an emergency fund"
Risk: "low"
"""

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_cohere import ChatCohere
from typing import TypedDict
import os

# Setup Cohere
api_key_prod = "uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr"
llm = ChatCohere(model="command-r", temperature=0.4,cohere_api_key=api_key_prod)

# 0. Define state schema
class FinanceState(TypedDict, total=False):
    step: str
    goal: str
    risk_profile: str
    suggestion: str

# 1. Greet
def greet(state):
    res = llm.invoke("You are a financial advisor bot. Start the session with a greeting.")
    print(f"🤖 {res.content}")
    return {"step": "ask_goal"}

# 2. Ask goal
# def ask_goal(state):
#     res = llm.invoke("Ask the user what financial goal they want to achieve.")
#     print(f"🎯 {res.content}")
#     goal = input("User: ")
#     return {"goal": goal, "step": "ask_risk"}

# # 3. Ask risk profile
# def ask_risk(state):
#     res = llm.invoke("Ask user their risk appetite: low, moderate, or high.")
#     print(f"📊 {res.content}")
#     risk = input("User: ")
#     return {"risk_profile": risk, "step": "suggest_plan"}

def ask_goal(state):
    if "goal" in state:
        return {"goal": state["goal"], "step": "ask_risk"}
    goal = input("Enter financial goal: ")
    return {"goal": goal, "step": "ask_risk"}

def ask_risk(state):
    if "risk_profile" in state:
        return {"risk_profile": state["risk_profile"], "step": "suggest_plan"}
    risk = input("Enter risk profile: ")
    return {"risk_profile": risk, "step": "suggest_plan"}


# 4. Suggest financial plan
def suggest_plan(state):
    prompt = (f"Suggest a financial plan for goal '{state['goal']}' with risk profile '{state['risk_profile']}' in 2-3 sentences."
              f"Suggest a personalized investment plan that includes clear, actionable advice. "
    f"Be sure to mention terms like 'education', 'low-risk', and 'mutual funds' if appropriate.")
    res = llm.invoke(prompt)
    print(f"💡 {res.content}")
    return {"suggestion": res.content}

# 5. Define graph
graph = StateGraph(FinanceState)
graph.add_node("greet", RunnableLambda(greet))
graph.add_node("ask_goal", RunnableLambda(ask_goal))
graph.add_node("ask_risk", RunnableLambda(ask_risk))
graph.add_node("suggest_plan", RunnableLambda(suggest_plan))

graph.set_entry_point("greet")
graph.add_edge("greet", "ask_goal")
graph.add_edge("ask_goal", "ask_risk")
graph.add_edge("ask_risk", "suggest_plan")
graph.add_edge("suggest_plan", END)

agent = graph.compile()

from IPython.display import Image, display
# display(Image(agent.get_graph().draw_mermaid_png()))

if __name__ == "__main__":
    print("💰 Finance Advisor Agent Ready")
    agent.invoke({})

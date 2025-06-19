from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
#from langchain_cohere import ChatCohere
from llm.oci_genai import initialize_llm
from typing import TypedDict, Dict, Any
import os
import time, re
from langchain.memory import ConversationBufferMemory
from langchain_cohere import ChatCohere
from langchain_core.runnables import RunnableLambda, RunnableParallel

#llm = initialize_llm()
api_key_prod = "uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr"
llm = ChatCohere(model="command-r-plus", temperature=0.3, cohere_api_key=api_key_prod)
# Define the agent state
class AgentState(TypedDict):
    input: str
    next: str
    output: str
    latency: float
    search_result: str
    math_result: str

# Initialize LLM and memory


memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Classify intent
def classify_intent(state: AgentState) -> Dict[str, Any]:
    query = state["input"]
    history_str = memory.load_memory_variables({})["history"]
    prompt = f"""You are a financial advisor.

Conversation so far:
{history_str}

Classify this client request:
"{query}"

Choose ONLY ONE of the following labels:
- plan
- math
- search
- memory
- fallback
"""

    result = llm.invoke(prompt).content.strip().lower()

    # Force mapping
    if "plan" in result:
        return {"next": "plan"}
    elif "math" in result or "tax" in result:
        return {"next": "math"}
    elif "search" in result or "define" in result:
        return {"next": "search"}
    elif "memory" in result or "remember" in result:
        return {"next": "memory"}
    else:
        return {"next": "fallback"}

# Tax calculator (old/new regime)
def tax_calculation_tool(state: AgentState) -> str:
    try:
        text = state["input"].lower()
        income_match = re.search(r"income\s*(\d+)", text)
        income = int(income_match.group(1)) if income_match else 0
        deduction_match = re.search(r"deduction\s*(\d+)", text)
        deduction = int(deduction_match.group(1)) if deduction_match else 150000
        taxable_income = max(0, income - deduction)
        regime = "new" if "new" in text else "old"

        tax = 0
        last_limit = 0
        slabs = [(250000, 0.0), (500000, 0.05), (1000000, 0.2), (float("inf"), 0.3)] if regime == "old" else [
            (300000, 0.0), (600000, 0.05), (900000, 0.1), (1200000, 0.15),
            (1500000, 0.2), (float("inf"), 0.3)
        ]

        for limit, rate in slabs:
            if taxable_income > limit:
                tax += (limit - last_limit) * rate
                last_limit = limit
            else:
                tax += (taxable_income - last_limit) * rate
                break

        cess = tax * 0.04
        total = tax + cess

        return (
            f"🧾 Tax Summary ({regime} regime):\n"
            f"- Income: ₹{income:,}\n"
            f"- Deduction: ₹{deduction:,}\n"
            f"- Taxable: ₹{taxable_income:,}\n"
            f"- Tax: ₹{tax:,.0f}\n"
            f"- Cess: ₹{cess:,.0f}\n"
            f"💰 Total: ₹{total:,.0f}"
        )
    except Exception as e:
        return f"Tax error: {e}"

# Retirement planner
def retirement_planner_tool(state: AgentState) -> Dict[str, str]:
    text = state["input"].lower()
    match_years = re.search(r"(\d+)\s*year", text)
    match_income = re.search(r"(\d{2,6})\s*(k|thousand|lakh|crore)?", text)
    years = int(match_years.group(1)) if match_years else 20
    monthly_income = int(match_income.group(1)) * 1000 if match_income else 45000
    inflation, returns = 0.06, 0.12
    future_income = monthly_income * ((1 + inflation) ** years)
    corpus = future_income * 12 * 25
    r = returns / 12
    n = years * 12
    sip = corpus * r / (((1 + r)**n - 1))
    return {
        "output": (
            f"📈 Retirement Planning:\n"
            f"- Target: ₹{monthly_income}/month in {years} years\n"
            f"- Future monthly: ₹{future_income:,.0f}\n"
            f"- Corpus: ₹{corpus/1e7:.2f} Cr\n"
            f"- SIP needed: ₹{sip:,.0f}/month"
        )
    }

# Finance concept lookup
def finance_search_tool(state: AgentState) -> str:
    topic = state["input"].lower()
    if "etf" in topic:
        return "📈 ETF = basket of assets traded on exchange like a stock."
    elif "cagr" in topic:
        return "📊 CAGR = Compound Annual Growth Rate."
    elif "inflation" in topic:
        return "💸 Inflation = reduction in money value over time."
    return "📚 No results, please rephrase your finance question."

# Memory recall
def memory_tool(state: AgentState) -> Dict[str, str]:
    history = memory.load_memory_variables({})["history"]
    query = state["input"].lower()
    user_msgs = [m.content for m in history if m.type == "human"]
    agent_msgs = [m.content for m in history if m.type == "ai"]

    if not user_msgs:
        return {"output": "I don't recall any previous questions."}

    if "goal" in query or "achieve" in query:
        for i in reversed(range(len(user_msgs))):
            if "goal" in user_msgs[i].lower() or "retirement" in user_msgs[i].lower():
                # Return the agent response to that goal
                if i < len(agent_msgs):
                    return {"output": f"Here's how you can achieve your goal:\n{agent_msgs[i]}"}
        return {"output": "I couldn't find any goal you previously asked about."}

    if "first" in query:
        return {"output": f"You first asked: {user_msgs[0]}"}

    return {"output": f"You previously asked: {user_msgs[-1]}"}


# Fallback handler
def fallback_tool(state: AgentState) -> Dict[str, str]:
    return {"output": "I didn't understand. Please clarify."}

# Prefetch both tools
def prefetch_with_merge(state: AgentState) -> Dict[str, Any]:
    tools = RunnableParallel({
        "search_result": RunnableLambda(finance_search_tool),
        "math_result": RunnableLambda(tax_calculation_tool)
    })
    return tools.invoke(state)

# Response generator
def generate_response(state: AgentState) -> Dict[str, str]:
    if state["next"] == "math":
        return {"output": state.get("math_result", "No math output")}
    elif state["next"] == "search":
        return {"output": state.get("search_result", "No search output")}
    else:
        return {"output": "Unexpected"}

# LangGraph setup
builder = StateGraph(AgentState)
builder.add_node("classify", RunnableLambda(classify_intent))
builder.add_node("prefetch", RunnableLambda(prefetch_with_merge))
builder.add_node("respond", RunnableLambda(generate_response))
builder.add_node("plan", RunnableLambda(retirement_planner_tool))
builder.add_node("memory", RunnableLambda(memory_tool))
builder.add_node("fallback", RunnableLambda(fallback_tool))

builder.set_entry_point("classify")
builder.add_conditional_edges("classify", lambda s: s["next"], {
    "plan": "plan",
    "math": "prefetch",
    "search": "prefetch",
    "memory": "memory",
    "fallback": "fallback"
})
builder.add_edge("prefetch", "respond")
builder.add_edge("respond", END)
builder.add_edge("plan", END)
builder.add_edge("memory", END)
builder.add_edge("fallback", END)

graph = builder.compile()

# Sample queries
queries = [
    "Income 1200000 old regime",
    "Calculate tax for income 850000 deduction 200000 new regime",
    "Goal: Plan for retirement in 20 years with monthly income of 45000",
    "What is CAGR?",
    "What was my first question?",
    "Tell me again how do I achive my Goal"
]

# Run agent
print("\n🧾 Finance Agent Results:\n" + "-"*30)
for query in queries:
    start = time.time()
    result = graph.invoke({
        "input": query,
        "search_result": "",
        "math_result": "",
        "next": "",
        "output": "",
        "latency": 0.0
    })
    end = time.time()
    memory.save_context({"input": query}, {"output": result["output"]})
    print(f"\n💬 {query}")
    print(f"🤖 {result['output']}")
    print(f"⏱ {round(end - start, 3)} sec")


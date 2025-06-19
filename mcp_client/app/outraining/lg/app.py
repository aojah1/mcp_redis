import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_cohere import ChatCohere
from typing import TypedDict

# === Setup Cohere ===
api_key_prod = "uUlulV3HkN4ti01lrNIS6rwYgHoPkKInUoWVLBjr"
llm = ChatCohere(model="command-r", temperature=0.4, cohere_api_key=api_key_prod)

# === Define Shared State ===
class FinanceState(TypedDict, total=False):
    step: str
    goal: str
    risk_profile: str
    income: str
    timeline_estimate: str
    tax_advice: str
    suggestion: str

# === Agent Functions ===
def goal_clarifier(state):
    st.subheader("🎯 Step 1: Define Your Financial Goal")
    res = llm.invoke("You are GoalClarifierAgent. Ask the user their primary financial goal. Present 2-3 example goals.")
    st.info(res.content)
    with st.form("goal_form"):
        goal = st.text_input("What's your primary financial goal?", key="goal")
        submitted = st.form_submit_button("Submit Goal")
    if submitted and goal:
        return {"goal": goal, "step": "risk_assessor"}

def risk_assessor(state):
    st.subheader("📊 Step 2: Risk Appetite")
    res = llm.invoke("You are RiskAssessorAgent. Ask the user to describe their risk appetite: low, moderate, or high.")
    st.info(res.content)
    with st.form("risk_form"):
        risk = st.radio("What is your risk appetite?", ["Low", "Moderate", "High"], key="risk_profile")
        submitted = st.form_submit_button("Submit Risk Profile")
    if submitted:
        return {"risk_profile": risk.lower(), "step": "budget_analyzer"}

def budget_analyzer(state):
    st.subheader("💵 Step 3: Monthly Income / Investment")
    res = llm.invoke("You are BudgetAnalyzerAgent. Ask the user for their monthly income or savings available.")
    st.info(res.content)
    with st.form("income_form"):
        income = st.text_input("Monthly income or investment amount (USD):", key="income")
        submitted = st.form_submit_button("Submit Income")
    if submitted and income:
        return {"income": income, "step": "summary_display"}

def timeline_estimator(state):
    prompt = (
        f"You are TimelineEstimatorAgent. Estimate how long it would take to achieve the goal '{state['goal']}' "
        f"with an income of '{state['income']}' and risk profile '{state['risk_profile']}'. Keep it brief (2–3 lines)."
    )
    res = llm.invoke(prompt)
    return {"timeline_estimate": res.content}

def tax_planner(state):
    prompt = (
        f"You are TaxPlannerAgent. Provide New York-specific tax advice for the goal '{state['goal']}' "
        f"with income '{state['income']}' and risk '{state['risk_profile']}'. Keep it brief (2–3 lines)."
    )
    res = llm.invoke(prompt)
    return {"tax_advice": res.content}

def plan_recommender(state):
    prompt = (
        f"You are PlanRecommenderAgent. Based on goal '{state['goal']}', risk '{state['risk_profile']}', "
        f"income '{state['income']}', timeline '{state['timeline_estimate']}', and tax advice '{state['tax_advice']}', "
        f"provide a 2–3 sentence plan."
    )
    res = llm.invoke(prompt)
    return {"suggestion": res.content}

# === UI Setup ===
st.set_page_config(page_title="AI Financial Planner", layout="centered")
st.title("💼 AI-Powered Financial Planning Assistant")
st.markdown("Use this smart assistant to generate a personalized financial strategy based on your goals, risk profile, and income.")

# === Session Initialization ===
if "step" not in st.session_state:
    st.session_state.step = "goal_clarifier"
    st.session_state.data = {}

# === Main Flow ===
current_step = st.session_state.step
state_data = st.session_state.data

step_funcs = {
    "goal_clarifier": goal_clarifier,
    "risk_assessor": risk_assessor,
    "budget_analyzer": budget_analyzer,
}

if current_step in step_funcs:
    result = step_funcs[current_step](state_data)
    if result:
        st.session_state.data.update(result)
        st.session_state.step = result["step"]
        st.experimental_rerun()

elif current_step == "summary_display":
    with st.spinner("Generating your personalized financial plan..."):
        # Run parallel agents
        timeline = timeline_estimator(state_data)
        tax = tax_planner(state_data)

        st.session_state.data.update(timeline)
        st.session_state.data.update(tax)

        # Final recommendation
        recommendation = plan_recommender(st.session_state.data)
        st.session_state.data.update(recommendation)

        st.session_state.step = "done"
        st.experimental_rerun()

elif current_step == "done":
    st.success("✅ Financial planning complete!")

    # Display results in columns for clarity
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Timeline Estimate")
        st.markdown(st.session_state.data["timeline_estimate"])

    with col2:
        st.subheader("🧾 Tax Advice")
        st.markdown(st.session_state.data["tax_advice"])

    st.divider()

    st.subheader("💡 Personalized Recommendation")
    st.markdown(st.session_state.data["suggestion"])

    if st.button("🔁 Start Over"):
        st.session_state.clear()
        st.experimental_rerun()

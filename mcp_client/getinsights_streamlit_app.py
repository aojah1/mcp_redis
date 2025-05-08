
import streamlit as st
import asyncio
from collections import deque
from langchain_core.messages import HumanMessage, AIMessage
import nest_asyncio
nest_asyncio.apply()

# Import the graph builder from your module
from redis_langgraph_supervisor import main as initialize_agent

st.title("🔍 GetInsights Supervisor Chatbot")

if 'agent' not in st.session_state:
    st.session_state.agent = asyncio.run(initialize_agent())
if 'history' not in st.session_state:
    st.session_state.history = deque(maxlen=30)

user_input = st.text_input("Ask a question to the supervisor:", "")

if user_input:
    st.session_state.history.append(HumanMessage(content=user_input))
    with st.spinner("Thinking..."):
        result = asyncio.run(st.session_state.agent.ainvoke({"messages": list(st.session_state.history)}))
        ai_msg = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        reply = ai_msg.content if ai_msg else "⚠️ (no reply)"
        st.session_state.history.append(AIMessage(content=reply))
        st.markdown(f"**🤖 Response:** {reply}")

# Display chat history
if st.session_state.history:
    st.markdown("### Chat History")
    for msg in st.session_state.history:
        role = "🧑‍💼 You" if isinstance(msg, HumanMessage) else "🤖 AI"
        st.markdown(f"**{role}:** {msg.content}")

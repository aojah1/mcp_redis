import streamlit as st
import asyncio, sys
from pathlib import Path
import nest_asyncio

# 🔧 Inject project root
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio, os
from getinsights.api_getinsights import invoke


def ask_insight(prompt: str) -> str:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(invoke(stream_mode="messages", prompt= prompt))

# Streamlit UI
st.title("GetInsights Test Harness")

user_input = st.text_input("Enter your question:")
if st.button("Get Insight"):
    if user_input:
        with st.spinner("Processing..."):
            answer = ask_insight(user_input)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question.")

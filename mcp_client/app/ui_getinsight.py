import streamlit as st
import asyncio, sys
from pathlib import Path
import nest_asyncio

# 🔧 Inject project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio, os
from mcp_client.app.api_getinsights import search, client


async def invoke(prompt):
    input_message = HumanMessage(content=prompt)
    thread = await client.threads.create()
    print(f"ThreadId: '{thread['thread_id']}'")
    assistant_id = await search()

    last_content = None
    async for part in client.runs.stream(
        thread["thread_id"],
        assistant_id=assistant_id,
        input={"messages": [input_message]},
        stream_mode="messages"
    ):
        event_type, data_list = part
        if isinstance(data_list, list):
            for item in data_list:
                if "content" in item:
                    last_content = item["content"]
    return last_content or "[No content found]"

def ask_insight(prompt: str) -> str:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(invoke(prompt))

# Streamlit UI
st.title("GetInsights GUI")

user_input = st.text_input("Enter your question:")
if st.button("Get Insight"):
    if user_input:
        with st.spinner("Processing..."):
            answer = ask_insight(user_input)
        st.subheader("Answer:")
        st.write(answer)
    else:
        st.error("Please enter a question.")

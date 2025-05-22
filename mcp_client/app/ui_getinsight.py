import streamlit as st
import asyncio, sys
from pathlib import Path

# 🔧 Inject project root
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio
from pprint import pprint
from mcp_client.app.api_getinsights import search

URL = "http://127.0.0.1:2024"
client = get_client(url=URL)


async def invoke(prompt):
    input_message = HumanMessage(content=prompt)

    thread = await client.threads.create()
    print(f"ThreadId: '{thread['thread_id']}'")

    last_content = None  # 🔧 Track last message

    async for part in client.runs.stream(
        thread["thread_id"],
        assistant_id=await search(),
        input={"messages": [input_message]},
        stream_mode="messages"
    ):
        event_type, data_list = part
        if isinstance(data_list, list):
            for item in data_list:
                if "content" in item:
                    last_content = item["content"]

    return last_content or "[No content found]"  # ✅ Return result


# Synchronous wrapper for Streamlit
def ask_insight(prompt: str) -> str:
    return asyncio.run(invoke(prompt))

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

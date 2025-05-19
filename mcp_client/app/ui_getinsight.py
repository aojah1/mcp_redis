import streamlit as st
import asyncio, sys
from pathlib import Path
# ensure project root is in sys.path so 'mcp_client' can be imported
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the correct build_graph from your agent module
from mcp_client.app.askdata_getinsights import build_graph  # ensure this returns your compiled StateGraph
from langchain_core.messages import HumanMessage

async def ask_insight_async(prompt: str):
    # Build (or reuse) the graph
    graph = await build_graph()
    # Invoke with a single HumanMessage
    output = await graph.ainvoke({"messages": [HumanMessage(content=prompt)]})
    # Return the last AIMessage content
    return output["messages"][-1].content

# Synchronous wrapper for Streamlit
def ask_insight(prompt: str) -> str:
    return asyncio.run(ask_insight_async(prompt))

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

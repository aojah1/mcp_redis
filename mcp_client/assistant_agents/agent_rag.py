from mcp_client.tools.tool_rag import rag_agent_service
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
#from langgraph.prebuilt import create_react_agent
import re
from typing import Optional
#from langgraph.prebuilt import ToolNode
#from langgraph.prebuilt import tools_condition
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

# Initialize the ReAct agent manually

######### RAG Agent ###########


def rag_agent(llm:BaseModel):
    return create_react_agent(
        model=llm,
        tools=[rag_agent_service],
        name="rag_expert",
        prompt="You are a world class RAG expert with access to unstructured vector data. Only use the provided tool for this type of search. Don't search the internet or reply by yourself.",
    )

class State(BaseModel):
    messages: list[BaseMessage]
    next: Optional[str] = None

def rag_node(state: State, llm: BaseModel):
    # 1) build the agent
    agent = rag_agent(llm)

    # 2) extract the last user message
    messages = state.get("messages", [])
    if not messages:
        raise ValueError("No messages in state")
    user_input = messages[-1].content
    print("User Input Received:", user_input)

    # 3) invoke the agent with a proper `messages` list
    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })
    print("Raw Agent Response:", result)

    # 4) pull out the final AIMessage
    msgs = result.get("messages", [])
    ai_msg = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    if ai_msg is None:
        raise RuntimeError(f"No AIMessage in response: {msgs!r}")

    # 5) return just the text
    text = ai_msg.content

    # LangGraph wants a dict here:
    return {
        "messages": state["messages"] + [AIMessage(content=text)]
    }


# Test Cases -
# now invoke the tool with the “state” envelope:
def test_case():
    from mcp_client.llm.oci_genai import initialize_llm

    raw_state = {
        "messages": [HumanMessage(content="how to create a good recipe")]
    }

    answer = rag_node(raw_state, initialize_llm())
    print("Final Answer:", answer)

if __name__ == "__main__":
    test_case()

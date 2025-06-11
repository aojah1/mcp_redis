#!/usr/bin/env python3.13
import uuid
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph         import StateGraph, END
from langgraph.prebuilt      import ToolNode, tools_condition, create_react_agent
from langgraph_swarm         import SwarmState, create_handoff_tool, add_active_agent_router
from llm.oci_genai           import initialize_llm

from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver


# ────────────────────────────────────────────────────────
# 0) bootstrap env
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ────────────────────────────────────────────────────────
# 1) your hand-off tools—these know how to mutate `state` and return a simple string
transfer_to_bob = create_handoff_tool(
    agent_name="Bob",
    description="Transfer the conversation to Bob, the pirate expert."
)
transfer_to_alice = create_handoff_tool(
    agent_name="Alice",
    description="Transfer the conversation back to Alice, the addition expert."
)

# ────────────────────────────────────────────────────────
# 2) build your agents, passing in the hand-off tools
model = initialize_llm()  # OCI/Cohere

def add(a: int, b: int) -> int:
    """
    add two numbers
    :param a:
    :param b:
    :return:
    """

    return a + b

alice = create_react_agent(
    model,
    tools=[add, transfer_to_bob],
    prompt="""
You are Alice, an addition expert.

If the user asks to speak to Bob (e.g. mentions 'Bob' or 'pirate'),
you MUST call the tool `transfer_to_bob` (no other tool calls).
Otherwise, answer addition questions normally.
""",
    name="Alice",
)

bob = create_react_agent(
    model,
    tools=[transfer_to_alice],
    prompt="""
You are Bob, a pirate expert.

If the user asks to go back to Alice (e.g. mentions 'Alice' or 'math'),
you MUST call the tool `transfer_to_alice`.
Otherwise, reply in pirate-speak.
""",
    name="Bob",
)

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages      import BaseMessage

class CleanSaver(InMemorySaver):
    def put_writes(self, writes: dict, *args, **kwargs):
        def scrub(obj):
            if isinstance(obj, BaseMessage):
                # reduce every Message to just its role/content
                return {"role": obj.role, "content": obj.content}
            elif isinstance(obj, dict):
                return {k: scrub(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [scrub(v) for v in obj]
            else:
                return obj

        clean = scrub(writes)
        return super().put_writes(clean, *args, **kwargs)


# ────────────────────────────────────────────────────────
# 3) manual StateGraph wiring
def bob_the_pirate():
    # Store for long-term (across-thread) memory
    #across_thread_memory = InMemoryStore()

    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()
    #checkpointer = CleanSaver()

    wf = StateGraph(SwarmState)

    # register our two experts
    wf.add_node("Alice", alice)
    wf.add_node("Bob",   bob)

    # single ToolNode for both hand-off tools
    wf.add_node(
        "tool",
        ToolNode(tools=[transfer_to_bob, transfer_to_alice])
    )

    # if an AIMessage emits a tool_call, run the tool; else END
    wf.add_conditional_edges("Alice", tools_condition, ["tool", END])
    wf.add_conditional_edges("Bob",   tools_condition, ["tool", END])

    # after END, router reads state.active_agent and continues there
    wf = add_active_agent_router(
        builder=wf,
        route_to=["Alice", "Bob"],
        default_active_agent="Alice",
    )

    app = wf.compile()
    return app

# ────────────────────────────────────────────────────────
# 4) synchronous REPL loop
from langchain_core.messages import HumanMessage, AIMessage

def get_data():
    app    = bob_the_pirate()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # this will hold the full conversation history
    history = []

    print("🔧 Swarm ready — type 'exit' to quit\n")
    try:
        while True:
            text = input("❓> ").strip()
            if not text or text.lower() in {"exit", "quit"}:
                break

            # 1) add the new user turn
            history.append(HumanMessage(content=text))

            # 2) invoke with the entire history
            output = app.invoke(
                {"messages": history},
                config
            )

            # 3) pull out the AI's reply
            reply = next(
                (m for m in reversed(output["messages"]) if isinstance(m, AIMessage)),
                None
            )
            if reply:
                print("→ AI says:", reply.content)
                # 4) append it to history so it goes back in next turn
                history.append(reply)
            else:
                print("→ (no AI reply found)")

    finally:
        if hasattr(app, "_close"):
            try: app._close()
            except TypeError: pass


if __name__ == "__main__":
    get_data()

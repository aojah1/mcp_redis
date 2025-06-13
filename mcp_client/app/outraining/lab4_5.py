#%% md
# Lang-graph simulated workflow travel planner
import uuid
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda


# 1. Define node functions
def greet(state):
    print("🤖 Hello! I'm your travel planner.")
    return {"step": "ask_destination"}

def ask_destination(state):
    destination = input("🌍 Where do you want to travel? ")
    return {"destination": destination, "step": "get_weather"}

def get_weather(state):
    destination = state["destination"]
    print(f"📡 Fetching weather for {destination}...")
    # Simulated weather
    return {"weather": "sunny", "step": "plan_trip"}

def plan_trip(state):
    weather = state["weather"]
    destination = state["destination"]
    print(f"🧳 Planning trip to {destination} with {weather} weather.")
    return {"plan": f"Pack light clothes for {destination}. Enjoy the sun!"}

# 2. Define the graph
from typing import TypedDict

# 2. Define state schema
class TravelState(TypedDict, total=False):
    step: str
    destination: str
    weather: str
    plan: str

graph = StateGraph(TravelState)
from langgraph.checkpoint.memory import MemorySaver
within_thread_memory = MemorySaver()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
# 3. Add nodes
graph.add_node("greet", RunnableLambda(greet))
graph.add_node("ask_destination", RunnableLambda(ask_destination))
graph.add_node("get_weather", RunnableLambda(get_weather))
graph.add_node("plan_trip", RunnableLambda(plan_trip))

# 4. Add edges
graph.set_entry_point("greet")
graph.add_edge("greet", "ask_destination")
graph.add_edge("ask_destination", "get_weather")
graph.add_edge("get_weather", "plan_trip")
graph.add_edge("plan_trip", END)

# 5. Compile the graph
agent = graph.compile(checkpointer=within_thread_memory)


# 6. Run the agent
initial_state = {}
agent.invoke(initial_state, config)

#print(agent.get_graph())


#from IPython.display import Image, display
#display(Image(agent.get_graph().draw_mermaid_png()))

################## Task 2


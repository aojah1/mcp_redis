import oci,os, re
import uuid
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from oci.addons.adk import Agent, AgentClient, tool
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, Optional, Literal
from langgraph.checkpoint.memory import MemorySaver

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file(profile_name=CONFIG_PROFILE)  # Update this with your own profile name
sess_id = ""

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_EP_ID_2 = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqa4hiv7nfbfmp65gwcuxbuncjovhtzx74rjfvbyedqxf6q"
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")



# === Step 1: Register Tools with OCI ADK ===
@tool
def get_employee_info(employee_id: str):
    """
    Fetch basic employee info like name and department.
    """
    db = {
        "E123": {"name": "Alice", "department": "Finance"},
        "E456": {"name": "Bob", "department": "HR"},
        "E789": {"name": "Carol", "department": "IT"}
    }
    return db.get(employee_id, {"error": "Employee not found"})

@tool
def get_hr_policy(department: str):
    """
    Fetch department-specific HR policy.
    """
    policies = {
        "Finance": "Submit receipts within 10 days.",
        "HR": "Complete annual compliance training.",
        "IT": "Remote-first policy for engineers."
    }
    return {"policy": policies.get(department, "No policy available.")}

# === Step 2: Initialize OCI Agent ===
client = AgentClient(
    auth_type="api_key",
    profile=CONFIG_PROFILE,
    region="us-chicago-1"
)

agent = Agent(
    client=client,
    agent_endpoint_id=AGENT_EP_ID,
    instructions="You have access to employee_info and hr_policy tools. Use them when instructed.",
    tools=[get_employee_info, get_hr_policy]
)

agent.setup()  # Only required once to register tools

# === Step 3: Define LangGraph State ===
class AgentState(TypedDict):
    employee_id: str
    name: Optional[str]
    department: Optional[str]
    policy: Optional[str]
    error: Optional[str]
    step: Literal["fetch_info", "fetch_policy", "done"]

# === Step 4: LangGraph Nodes Using Remote Tool Calls ===
import json

def fetch_employee_info(state: AgentState) -> AgentState:
    prompt = f"""
You are a tool execution agent. Use `get_employee_info` to retrieve details for employee ID: {state['employee_id']}.
Only return a JSON object in this format (no extra explanation):

{{
  "name": "Bob",
  "department": "HR"
}}
"""

    response = agent.run(prompt)  # OCI Agent called and Tool utilization starts Tool : get_employee_info
    result_text = response.data["message"]["content"]["text"]

    try:
        # load as valid JSON
        result_dict = json.loads(result_text)
    except json.JSONDecodeError:
        try:
            # Fix single quotes
            result_dict = json.loads(result_text.replace("'", '"'))
        except Exception:
            return {
                "error": "Unable to parse employee info",
                "employee_id": state["employee_id"],
                "step": "done"
            }

    # check for required keys
    if not all(k in result_dict for k in ("name", "department")):
        return {
            "error": f"Incomplete info: {result_dict}",
            "employee_id": state["employee_id"],
            "step": "done"
        }

    return {
        "employee_id": state["employee_id"],
        "name": result_dict["name"],
        "department": result_dict["department"],
        "step": "fetch_policy"
    }



def fetch_hr_policy(state: AgentState) -> AgentState:
    prompt = f"""
Use the `get_hr_policy` tool for department: {state['department']}.
Return only a JSON object with one key 'policy', like:

{{ "policy": "Some HR policy text" }}
"""

    response = agent.run(prompt) # OCI Agent called and Tool utilization starts Tool get_hr_policy
    result_text = response.data["message"]["content"]["text"]

    try:
        result_dict = json.loads(result_text)
    except json.JSONDecodeError:
        try:
            result_dict = json.loads(result_text.replace("'", '"'))
        except Exception:
            return {
                **state,
                "error": "Unable to parse policy response",
                "step": "done"
            }

    if "policy" not in result_dict:
        return {
            **state,
            "error": f"Missing 'policy' in result: {result_dict}",
            "step": "done"
        }

    return {
        **state,
        "policy": result_dict["policy"],
        "step": "done"
    }


# === Step 5: Build LangGraph ===
checkpointer = MemorySaver()
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
builder = StateGraph(AgentState)
builder.add_node("fetch_info", RunnableLambda(fetch_employee_info))
builder.add_node("fetch_policy", RunnableLambda(fetch_hr_policy))

builder.set_entry_point("fetch_info")
builder.add_edge("fetch_info", "fetch_policy")
builder.add_edge("fetch_policy", END)
graph = builder.compile(checkpointer=checkpointer)

# === Step 6: Run the LangGraph with ADK Tool Calls ===
def run_fully_remote_demo(user_input: str):
    employee_id = next((w for w in user_input.split() if w.upper().startswith("E")), None)
    if not employee_id:
        print("Employee ID not found.")
        return

    result = graph.invoke({"employee_id": employee_id, "step": "fetch_info"}, config)

    if result.get("error"):
        print(f"{result['error']}")
    else:
        print(f"✅ Employee: {result['name']} ({result['employee_id']})")
        print(f"🏢 Department: {result['department']}")
        print(f"📄 Policy: {result['policy']}")

# === CLI ===
if __name__ == "__main__":
    query = input("Enter your request (e.g., 'What is policy for E456?'): ")
    run_fully_remote_demo(query)
    #run_fully_remote_demo("Fetch employee info again for the id provided earlier")
import oci,os
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any
from oci.addons.adk import Agent, AgentClient, tool

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file(profile_name=CONFIG_PROFILE)  # Update this with your own profile name
sess_id = ""

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")

# === User-Defined Tools ===

@tool
def get_employee_info(employee_id: str) -> Dict[str, Any]:
    """
    Retrieves basic employee information.
    Args:
      employee_id (str): Unique employee identifier.
    Returns:
      Dict with employee details.
    """
    # In production, connect to your employee database or API.
    dummy_data = {
        "E123": {"name": "Anita Singh", "department": "Finance", "location": "Mumbai"},
        "E124": {"name": "Ravi Patel", "department": "IT", "location": "Bangalore"},
    }
    return dummy_data.get(employee_id, {"error": "Employee not found."})

@tool
def calculate_leave_balance(employee_id: str) -> Dict[str, Any]:
    """
    Returns the current leave balance for an employee.
    Args:
      employee_id (str): Employee code.
    Returns:
      Dict with leave balance details.
    """
    # Example: Replace with real logic/integration as needed.
    leave_db = {
        "E123": {"annual_leave": 7, "sick_leave": 2},
        "E124": {"annual_leave": 10, "sick_leave": 0},
    }
    return leave_db.get(employee_id, {"error": "No leave data found."})

@tool
def corporate_calculator(expression: str) -> Dict[str, str]:
    """
    Evaluates a mathematical expression (business calculations).
    Args:
      expression (str): Math formula, e.g. '2025 * 1.08'
    Returns:
      Dict with result.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return {"expression": expression, "result": str(result)}
    except Exception as e:
        return {"error": f"Invalid expression: {e}"}

# === Main Agent Setup and Run ===

def main():
    # Step 1: Authenticate and connect to OCI GenAI
    client = AgentClient(
        auth_type="api_key",
        profile=CONFIG_PROFILE,
        region="us-chicago-1",
    )

    # Step 2: Configure the agent with business instructions and custom tools
    agent = Agent(
        client=client,
        agent_endpoint_id=AGENT_EP_ID,
        instructions="You are an employee support agent. Use available tools to answer user queries related to employee info, leave balance, and business calculations.",
        tools=[get_employee_info, calculate_leave_balance, corporate_calculator],
    )
    agent.setup()  # Register tools and instructions with remote agent endpoint

    # Step 3: Sample user queries the agent will handle
    queries = [
        "Show me the leave balance for employee E123.",
        "Get employee info for E124.",
        "Calculate 12500 * 1.18 for GST calculation."
    ]

    # Step 4: Run agent and display outputs
    for input_query in queries:
        print(f"\nUser: {input_query}")
        response = agent.run(input_query)
        response.pretty_print()


if __name__ == "__main__":
    main()


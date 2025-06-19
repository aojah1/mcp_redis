#%% md
## Setup Tool - OCI RAG AGENT SERVICE
#The OCI RAG Agent Service is a pre-built service from Oracle cloud, that is designed to perform multi-modal
# augmented search against any pdf (with embedded tables and charts) or txt files.

import oci,os
from langchain_core.tools import tool
from oci.generative_ai_agent_runtime import GenerativeAiAgentRuntimeClient
from oci.generative_ai_agent_runtime.models import CreateSessionDetails
from tools.tool_rag import initialize_oci_genai_agent_service

from dotenv import load_dotenv
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = os.getenv("AGENT_EP_ID")
AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")

config = oci.config.from_file(profile_name="DEFAULT")  # Update this with your own profile name
sess_id = ""

llm_agent, llm_session= initialize_oci_genai_agent_service()
print("llm_agent" + str(llm_agent))
print("llm_session" + str(llm_session))


print(oci.__version__)

from typing import Dict
from oci.addons.adk import Agent, AgentClient, tool


@tool
def get_weather(location: str) -> Dict[str, str]:
    """
    Get the weather for a given location.

    Args:
      location(str): The location for which weather is queried
    """
    return {"location": location, "temperature": 72, "unit": "F"}


def main():
    client = AgentClient(
        auth_type="api_key",
        profile="DEFAULT",
        region="us-chicago-1",
    )

    agent = Agent(
        client=client,
        agent_endpoint_id=AGENT_EP_ID,
        instructions="You perform weather queries using tools.",
        tools=[get_weather]
    )

    agent.setup()

    input = "Is it cold in Seattle?"
    response = agent.run(input)

    # Print the response
    response.pretty_print()


if __name__ == "__main__":
    main()

#%% md
## Setup Tool - OCI RAG AGENT SERVICE
#The OCI RAG Agent Service is a pre-built service from Oracle cloud, that is designed to perform multi-modal
# augmented search against any pdf (with embedded tables and charts) or txt files.

import oci,os
from langchain_core.tools import tool
from oci.generative_ai_agent_runtime import GenerativeAiAgentRuntimeClient
from oci.generative_ai_agent_runtime.models import CreateSessionDetails

from dotenv import load_dotenv
from pathlib import Path

THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")
print(PROJECT_ROOT)
# Set up the OCI GenAI Agents endpoint configuration
AGENT_EP_ID = "ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqa4hiv7nfbfmp65gwcuxbuncjovhtzx74rjfvbyedqxf6q"

AGENT_SERVICE_EP = os.getenv("AGENT_SERVICE_EP")

config = oci.config.from_file(profile_name="DEFAULT")  # Update this with your own profile name
sess_id = ""


def initialize_oci_genai_agent_service():
    """Initialize OCI GenAI Agent Service and create a session"""

    # Initialize service client with default config file
    generative_ai_agent_runtime_client = oci.generative_ai_agent_runtime.GenerativeAiAgentRuntimeClient(
        config,
        service_endpoint=AGENT_SERVICE_EP)

    return generative_ai_agent_runtime_client

generative_ai_agent_runtime_client = initialize_oci_genai_agent_service()
print(generative_ai_agent_runtime_client)
import oci
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
        agent_endpoint_id="ocid1.genaiagentendpoint.oc1.us-chicago-1.amaaaaaawe6j4fqa4hiv7nfbfmp65gwcuxbuncjovhtzx74rjfvbyedqxf6q",
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

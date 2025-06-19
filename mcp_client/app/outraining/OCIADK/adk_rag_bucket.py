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

#%% md
### RAG : OCI ADK Agent + Knowledge base OCI Object Storage Buckets

object_storage = oci.object_storage.ObjectStorageClient(config)
namespace = object_storage.get_namespace().data
print("You can access your object storage bucket in namespace ", namespace)

import oci
import json
from oci.addons.adk import Agent, AgentClient, tool
from oci.object_storage import ObjectStorageClient

CONFIG_PROFILE = "DEFAULT"
BUCKET_NAME = "demo-agent-kb"
OBJECT_NAME = "policies.json"     # employees policies data file loaded in oci bucket

object_storage = ObjectStorageClient(config)

#object_storage.list_buckets(namespace_name, compartment_id)
# Tool creation to fetch data from bucket

@tool
def retrieve_policy(policy_name: str) -> dict:
    """Retrieve policy content by name."""

    # Connect to bucket
    response = object_storage.get_object(namespace, BUCKET_NAME, OBJECT_NAME)

    # read the file in bucket
    retrived_data = json.loads(response.data.content.decode("utf-8"))
    return {"content": retrived_data.get(policy_name, "Policy not found.")}

client = AgentClient(auth_type="api_key", profile=CONFIG_PROFILE, region="us-chicago-1")
agent = Agent(
    client=client,
        agent_endpoint_id=AGENT_EP_ID,
    instructions="You are a company policy assistant. Retrieve policies as requested.",
    tools=[retrieve_policy]
)
agent.setup()

response = agent.run("Show me the leave policy.")
print(response.data["message"]["content"]["text"])


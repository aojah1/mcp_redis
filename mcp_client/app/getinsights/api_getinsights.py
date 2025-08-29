from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

LANGRAPH_DEV = os.environ.get("LANGRAPH_DEV", "http://127.0.0.1:2024")
#URL = "http://64.181.208.129:2024"
client = get_client(url=LANGRAPH_DEV)
assistant_id = ""

# Search all hosted graphs
async def search():
    assistants = await client.assistants.search(graph_id= "askdata_getinsights")
    assistant_id = assistants[0]["assistant_id"] # Unique ID e.g 0468dc38-81bf-5b14-969d-81bd9f36e07d

    print(f"Assistant ID: '{assistant_id}'")
    return assistant_id


### Stream_mode=updates
async def invoke(stream_mode: str, prompt):
    input_message = HumanMessage(
        #content="which Invoice I should pay first based criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'"
        content=prompt,
    )

    # Create a thread
    thread = await client.threads.create()
    print(f"ThreadId: '{thread['thread_id']}'")

    last_content = None

    async for part in client.runs.stream(
            thread["thread_id"],
            assistant_id=await search(),
            input={"messages": [input_message]},
            stream_mode=stream_mode):

        event_type, data_list = part  # ✅ part is (event_type, [dict, dict, ...])

        if isinstance(data_list, list):
            for item in data_list:
                if "content" in item:
                    last_content = item["content"]  # ✅ store last one

    return last_content or "[No content found]"


if __name__ == '__main__':
    content="which Invoice I should pay first based criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'"

    asyncio.run(invoke("values",content))
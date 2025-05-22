from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

LANGRAPH_DEV = os.environ.get("LANGRAPH_DEV", "http://127.0.0.1:2024")
#URL = "http://64.181.208.129:2024"
client = get_client(url=LANGRAPH_DEV)
assistant_id = ""

# Search all hosted graphs
async def search():
    assistants = await client.assistants.search(graph_id= "askdata_getinsights")
    assistant_id = assistants[0]["assistant_id"]

    print(f"Assistant ID: '{assistant_id}'")
    return assistant_id

async def invoke():
    # Create a new thread
    thread = await client.threads.create()

    answer = await client.runs.wait(
        thread_id=thread["thread_id"],
        assistant_id= await search(),
        input={"messages": "which Invoice I should pay first based criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'"}
    )

    ai_reply = next(
        (m for m in reversed(answer["messages"]) if m.get("type") == "ai"),
        None
    )

    if ai_reply:
        print("\n🧠 Assistant Reply:\n")
        print(ai_reply["content"])  # just print the message content

        # Optionally, pretty-print entire AI message
        # pprint(ai_reply)  # or use:
        # print(json.dumps(ai_reply, indent=2))
    else:
        print("No AI message found.")



### Stream_mode=updates
async def invoke(stream_mode: str):
    input_message = HumanMessage(
        content="which Invoice I should pay first based criteria such as highest amount due and highest past due date for 'session:e5f6a932-6123-4a04-98e9-6b829904d27f'"
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

    print("\n🧠 Final streamed response part 1:\n")
    print(last_content if last_content else "[No content found]")

    input_message = HumanMessage(
        content="What was the criteria used for the recommendation?"
    )

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

    print("\n🧠 Final streamed response part 2:\n")
    print(last_content if last_content else "[No content found]")

    input_message = HumanMessage(
        content="what was the amount due on the invoice you reccommened to pay ? Just provide me the amount due and no other information."
    )

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

    print("\n🧠 Final streamed response part 3:\n")
    print(last_content if last_content else "[No content found]")


if __name__ == '__main__':
    asyncio.run(invoke("messages"))
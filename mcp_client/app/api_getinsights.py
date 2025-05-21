from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio
from pprint import pprint

URL = "http://127.0.0.1:2024"
client = get_client(url=URL)

assistant_id="f9142976-9614-4f5f-9793-045acd655238"

# Search all hosted graphs
async def search():
    assistants = await client.assistants.search()
    assistant_id = assistants[0]["assistant_id"]
    pprint(assistant_id)

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



if __name__ == '__main__':
    asyncio.run(invoke())
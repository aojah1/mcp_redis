# redis_assistant.py
import asyncio, sys
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from openai.types.responses import ResponseTextDeltaEvent
from collections import deque

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent                 # repo root
MAIN_FILE    = (PROJECT_ROOT / "src" / "main.py").resolve()

async def build_agent():
    """Launch the Redis MCP helper (src/main.py) and return the Agent."""
    server = MCPServerStdio(
        params={
            "command": sys.executable,          # venv’s python 3.13
            "args": [str(MAIN_FILE)],           # plain script, no uvicorn
            "env": {
                "REDIS_HOST": "127.0.0.1",
                "REDIS_PORT": "6379",
                # add auth env vars if you need them
            },
        }
    )
    await server.connect()

    agent = Agent(
        name="Redis Assistant",
        instructions=(
            "You are a helpful assistant capable of reading and writing to "
            "Redis. Store every question and answer in the Redis Stream "
            "app:logger."
        ),
        mcp_servers=[server],
    )

    return server, agent

# CLI interaction
async def cli(agent, max_history=30):
    print("🔧 Redis Assistant CLI — Ask me something (type 'exit' to quit):\n")
    conversation_history = deque(maxlen=max_history)

    while True:
        q = input("❓> ")
        if q.strip().lower() in {"exit", "quit"}:
            break

        if (len(q.strip()) > 0):
            # Format the context into a single string
            history = ""
            for turn in conversation_history:
                prefix = "User" if turn["role"] == "user" else "Assistant"
                history += f"{prefix}: {turn['content']}\n"

            context = f"Conversation history:/n{history.strip()} /n New question:/n{q.strip()}"
            result = Runner.run_streamed(agent, context)

            response_text = ""
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    print(event.data.delta, end="", flush=True)
                    response_text += event.data.delta
            print("\n")

            # Add the user's message and the assistant's reply in history
            #conversation_history.append({"role": "user", "content": q})
            #quitconversation_history.append({"role": "assistant", "content": response_text})


# Main entry point
async def main():
    server, agent = await build_agent()
    async with server:          # ← ensures connect / disconnect in 1 task
        await cli(agent)


if __name__ == "__main__":
    asyncio.run(main())

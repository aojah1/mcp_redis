# redis_assistant.py
import asyncio, sys, os
from pathlib import Path
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from openai.types.responses import ResponseTextDeltaEvent
from collections import deque
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# LLM from OCI GenAI Services - Config
from langchain_community.chat_models import ChatOCIGenAI
from langchain.prompts import PromptTemplate  # For creating prompts


THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent                 # repo root
MAIN_FILE    = (PROJECT_ROOT / "mcp_server" / "main.py").resolve()

# ────────────────────────────────────────────────────────────────
# Load environment variables from .env file
# ────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

# ────────────────────────────────────────────────────────
# 3) OCI GenAI configuration
# ────────────────────────────────────────────────────────
COMPARTMENT_ID = os.getenv("OCI_COMPARTMENT_ID")
ENDPOINT       = os.getenv("OCI_GENAI_ENDPOINT")
MODEL_ID       = os.getenv("OCI_GENAI_MODEL_ID")
PROVIDER       = os.getenv("PROVIDER")
AUTH_TYPE      = "API_KEY"
CONFIG_PROFILE = "DEFAULT"


def initialize_llm():
    return ChatOCIGenAI(
        model_id=MODEL_ID,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        provider=PROVIDER,
        model_kwargs={
            "temperature": 0.5,
            "max_tokens": 512,
            # remove any unsupported kwargs like citation_types
        },
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE,
    )

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
        llm=initialize_llm(),
        mcp_servers=[server],
    )
    #agent = create_react_agent(llm, mcp_server=[server])

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

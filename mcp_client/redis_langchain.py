# redis_assistant_langgraph.py
import asyncio, sys
from pathlib import Path
from collections import deque

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain.agents import AgentType, initialize_agent

# LLM from OCI GenAI Services - Config
from langchain_community.chat_models import ChatOCIGenAI
from langchain.prompts import PromptTemplate  # For creating prompts

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
MAIN_FILE    = (PROJECT_ROOT / "mcp_server" / "main.py").resolve()


# Set your OCI credentials
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaau6esoygdsqxfz6iv3u7ghvosfskyvd6kroucemvyr5wzzjcw6aaa"
AUTH_TYPE = "API_KEY" # The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
CONFIG_PROFILE = "DEFAULT"

# Service endpoint
endpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
#model_id = "meta.llama-3.3-70b-instruct"
model_id = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyanrlpnq5ybfu5hnzarg7jomak3q6kyhkzjsl4qj24fyoq"
# Create an OCI Cohere LLM instance
# initialize interface

# ────────────────────────────────────────────────────────────────
# 1. Compile the LangGraph agent (needs an open MCP session)
# ────────────────────────────────────────────────────────────────
async def build_agent(session: ClientSession):
    tools = await load_mcp_tools(session)

    llm = ChatOCIGenAI(
        model_id=model_id,
        service_endpoint=endpoint,
        compartment_id=COMPARTMENT_ID,
        provider="cohere",
        model_kwargs={
            "temperature": 1,
            "max_tokens": 600,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "top_k": 0,
            "top_p": 0.75
        },
        auth_type=AUTH_TYPE,
        auth_profile=CONFIG_PROFILE
    )

    return create_react_agent(llm, tools)


# ────────────────────────────────────────────────────────────────
# 2. Plain CLI loop (full reply, no chunk printing)
# ────────────────────────────────────────────────────────────────
async def cli(agent, max_history: int = 30):
    print("🔧  Redis Assistant — type 'exit' to quit\n")
    history = deque(maxlen=max_history)

    while True:
        q = input("❓> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        messages = list(history) + [{"role": "user", "content": q}]

        result = await agent.ainvoke({"messages": messages})
        assistant_msg = next(
            (m.content for m in result["messages"][::-1]
             if getattr(m, "type", getattr(m, "role", "")) in {"assistant", "ai"}),
            ""
        )
        print(f"\n🤖 {assistant_msg}\n")

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": assistant_msg})


# ────────────────────────────────────────────────────────────────
# 3. Main — keep helper & session alive for whole program lifetime
# ────────────────────────────────────────────────────────────────
async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MAIN_FILE)],
        env={
            "REDIS_HOST": "127.0.0.1",
            "REDIS_PORT": "6379",
            # "REDIS_PASSWORD": "mysecret",  # add if your Redis needs AUTH
        },
    )

    # Helper process + pipes
    async with stdio_client(server_params) as (reader, writer):
        # MCP session over those pipes
        async with ClientSession(reader, writer) as session:
            await session.initialize()           # handshake

            agent = await build_agent(session)   # compile LangGraph
            await cli(agent)                     # run the REPL


if __name__ == "__main__":
    asyncio.run(main())

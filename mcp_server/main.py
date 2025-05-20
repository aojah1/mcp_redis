import sys

from common.connection import RedisConnectionManager
from common.server import mcp
import tools.server_management
import tools.misc
import tools.redis_query_engine
import tools.hash
import tools.list
import tools.string
import tools.json
import tools.sorted_set
import tools.set
import tools.stream
import tools.pub_sub
from dotenv import load_dotenv
from pathlib import Path
from common.config import MCP_TRANSPORT

# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

# ────────────────────────────────────────────────────────
# 2) Start MCP Server
# ────────────────────────────────────────────────────────
class RedisMCPServer:
    def __init__(self):
        print("Starting the RedisMCPServer", file=sys.stderr)

    def run(self):
        print(f"🔧 Starting Redis MCP Server (transport={MCP_TRANSPORT})")
        mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    server = RedisMCPServer()
    server.run()

# Redis MCP Server
[![smithery badge](https://smithery.ai/badge/@redis/mcp-redis)](https://smithery.ai/server/@redis/mcp-redis)

<a href="https://glama.ai/mcp/servers/@redis/mcp-redis">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@redis/mcp-redis/badge" alt="Redis Server MCP server" />
</a>

## Overview
The Redis MCP Server is a **natural language interface** designed for agentic applications to efficiently manage and search data in Redis. It integrates seamlessly with **MCP (Model Content Protocol) clients**, enabling AI-driven workflows to interact with structured and unstructured data in Redis. Using this MCP Server, you can ask questions like:

- "Store the entire conversation in a stream"
- "Cache this item"
- "Store the session with an expiration time"
- "Index and search this vector"

## Features
- **Natural Language Queries**: Enables AI agents to query and update Redis using natural language.
- **Seamless MCP Integration**: Works with any **MCP client** for smooth communication.
- **Full Redis Support**: Handles **hashes, lists, sets, sorted sets, streams**, and more.
- **Search & Filtering**: Supports efficient data retrieval and searching in Redis.
- **Scalable & Lightweight**: Designed for **high-performance** data operations.

## Tools

This MCP Server provides tools to manage the data stored in Redis.

- `string` tools to set, get strings with expiration. Useful for storing simple configuration values, session data, or caching responses.
- `hash` tools to store field-value pairs within a single key. The hash can store vector embeddings. Useful for representing objects with multiple attributes, user profiles, or product information where fields can be accessed individually.
- `list` tools with common operations to append and pop items. Useful for queues, message brokers, or maintaining a list of most recent actions.
- `set` tools to add, remove and list set members. Useful for tracking unique values like user IDs or tags, and for performing set operations like intersection.
- `sorted set` tools to manage data for e.g. leaderboards, priority queues, or time-based analytics with score-based ordering.
- `pub/sub` functionality to publish messages to channels and subscribe to receive them. Useful for real-time notifications, chat applications, or distributing updates to multiple clients.
- `streams` tools to add, read, and delete from data streams. Useful for event sourcing, activity feeds, or sensor data logging with consumer groups support.
- `JSON` tools to store, retrieve, and manipulate JSON documents in Redis. Useful for complex nested data structures, document databases, or configuration management with path-based access.

Additional tools.

- `query engine` tools to manage vector indexes and perform vector search
- `server management` tool to retrieve information about the database

## Installation

Follow these instructions to install the server.

```sh
# Clone the repository
git clone https://github.com/aojah1/mcp_redis.git
cd mcp-redis

# Install dependencies using uv
uv venv
source .venv/bin/activate
uv sync
```

## Configuration

To configure this Redis MCP Server, consider the following environment variables:

| Name                    | Description                                               | Default Value |
|-------------------------|-----------------------------------------------------------|---------------|
| `REDIS_HOST`            | Redis IP or hostname                                      | `"127.0.0.1"` |
| `REDIS_PORT`            | Redis port                                                | `6379`        |
| `REDIS_USERNAME`        | Default database username                                 | `"default"`   |
| `REDIS_PWD`             | Default database password                                 | ""            |
| `REDIS_SSL`             | Enables or disables SSL/TLS                               | `False`       |
| `REDIS_CA_PATH`         | CA certificate for verifying server                       | None          |
| `REDIS_SSL_KEYFILE`     | Client's private key file for client authentication       | None          |
| `REDIS_SSL_CERTFILE`    | Client's certificate file for client authentication       | None          |
| `REDIS_CERT_REQS`       | Whether the client should verify the server's certificate | `"required"`  |
| `REDIS_CA_CERTS`        | Path to the trusted CA certificates file                  | None          |
| `REDIS_CLUSTER_MODE`    | Enable Redis Cluster mode                                 | `False`       |

## Integration with OpenAI Agents SDK

Integrate this MCP Server with the OpenAI Agents SDK. Read the [documents](https://openai.github.io/openai-agents-python/mcp/) to learn more about the integration of the SDK with MCP.
Integration with OCI GenAI Service coming soon (will start with Cohere as Llama models has issues with tool calling as of today)
Install the Python SDK.

```commandline
pip install openai-agents
```

Configure the OpenAI token:

```commandline
export OPENAI_API_KEY="<openai_token>"
```

And run the [application](mcp_client/redis_assistant.py).

```commandline
python3.13 mcp_client/redis_langchain.py
```


### Using with Docker

You can use a dockerized deployment of this server. You can either build your own image or use the official [Redis MCP Docker](https://hub.docker.com/r/mcp/redis) image.

If you'd like to build your own image, the Redis MCP Server provides a Dockerfile. Build this server's image with:

```commandline
docker build -t mcp-redis .
```


### Troubleshooting

You can troubleshoot problems by tailing the log file.

```commandline
tail -f ~/Library/Logs/Claude/mcp-server-redis.log
```

## Integration with VS Code

To use the Redis MCP Server with VS Code, you need:

1. Enable the [agent mode](https://code.visualstudio.com/docs/copilot/chat/chat-agent-mode) tools. Add the following to your `settings.json`:

```commandline
{
  "chat.agent.enabled": true
}
```

2. Add the Redis MCP Server configuration to your `mcp.json` or `settings.json`:

```commandline
// Example .vscode/mcp.json
{
  "servers": {
    "redis": {
      "type": "stdio",
      "command": "<full_path_uv_command>",
      "args": [
        "--directory",
        "<your_mcp_server_directory>",
        "run",
        "src/main.py"
      ],
      "env": {
        "REDIS_HOST": "<your_redis_database_hostname>",
        "REDIS_PORT": "<your_redis_database_port>",
        "REDIS_USERNAME": "<your_redis_database_username>",
        "REDIS_PWD": "<your_redis_database_password>",
      }
    }
  }
}
```

```commandline
// Example settings.json
{
  "mcp": {
    "servers": {
      "redis": {
        "type": "stdio",
        "command": "<full_path_uv_command>",
        "args": [
          "--directory",
          "<your_mcp_server_directory>",
          "run",
          "src/main.py"
        ],
        "env": {
          "REDIS_HOST": "<your_redis_database_hostname>",
          "REDIS_PORT": "<your_redis_database_port>",
          "REDIS_USERNAME": "<your_redis_database_username>",
          "REDIS_PWD": "<your_redis_database_password>",
        }
      }
    }
  }
}
```

For more information, see the [VS Code documentation](https://code.visualstudio.com/docs/copilot/chat/mcp-servers).


## Testing

You can use the [MCP Inspector](https://modelcontextprotocol.io/docs/tools/inspector) for visual debugging of this MCP Server.

```sh
npx @modelcontextprotocol/inspector uv run src/main.py
```

## Example Use Cases
- **AI Assistants**: Enable LLMs to fetch, store, and process data in Redis.
- **Chatbots & Virtual Agents**: Retrieve session data, manage queues, and personalize responses.
- **Data Search & Analytics**: Query Redis for **real-time insights and fast lookups**.
- **Event Processing**: Manage event streams with **Redis Streams**.

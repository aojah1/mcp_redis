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
cd mcp_redis

# Optional commands
How to actually get Python 3.13 on macOS (change it for your machine)
    1 Homebrew (simplest)
    bash
    
    Edit
    brew update
    brew install python@3.13          # puts python3.13 in /opt/homebrew/bin
    echo 'export PATH="/opt/homebrew/opt/python@3.13/bin:$PATH"' >> ~/.zshrc
    exec $SHELL                       # reload shell so python3.13 is found
    python3.13 --version              # → Python 3.13.x
    2 pyenv (lets you switch versions)
    bash
    
    Edit
    brew install pyenv
    pyenv install 3.13.0
    pyenv global 3.13.0
    python --version                  # now 3.13.0

# Install dependencies
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .
```
# Install MCP Client requirement
 python3.13 -m pip install -r requirements.txt

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


Configure to use OCI GenAI Service as a next step:
Used Cohere Model for ReAct

```configure ~/.oci/config
https://docs.oracle.com/en-us/iaas/Content/API/Concepts/sdkconfig.htm
```
Use OCI Resource Principle for production

And run the [application](mcp_client/redis_langchain.py).

```commandline
python3.13 mcp_client/redis_langchain.py
```
### To Test the Graph in Dev/Local run the following, you will get an IP/Port that can be used to connect from a lagraph client
langgraph dev --config langgraph.json --allow-blocking

### To Run the graph using SSE transport - this will open an hhtp:port for the MCP client to listen in
python3.13 mcp_server/main.py
### on another terminal run - 
python3.13 mcp_client/redis_langchain.py


### Using with Docker

You can use a dockerized deployment of this server. You can either build your own image or use the official [Redis MCP Docker](https://hub.docker.com/r/mcp/redis) image.

If you'd like to build your own image, the Redis MCP Server provides a Dockerfile. Build this server's image with:

```commandline
docker build -t mcp_redis .
```


### Troubleshooting

You can troubleshoot problems by tailing the log file.

```commandline
tail -f ~/Library/Logs/mcp/mcp-server-redis.log
```

## Example Use Cases
- **AI Assistants**: Enable LLMs to fetch, store, and process data in Redis.
- **Chatbots & Virtual Agents**: Retrieve session data, manage queues, and personalize responses.
- **Data Search & Analytics**: Query Redis for **real-time insights and fast lookups**.
- **Event Processing**: Manage event streams with **Redis Streams**.

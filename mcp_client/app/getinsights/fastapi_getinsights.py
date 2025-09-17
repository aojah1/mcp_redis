from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, constr

from langgraph_sdk import get_client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, status
from fastapi.responses import JSONResponse


# ────────────────────────────────────────────────────────
# 1) bootstrap paths + env + llm
# ────────────────────────────────────────────────────────
THIS_DIR     = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # expects OCI_ vars in .env

LANGRAPH_DEV = os.environ.get("LANGRAPH_DEV", "http://127.0.0.1:2024")
#URL = "http://64.181.208.129:2024"
client = get_client(url=LANGRAPH_DEV)
assistant_id = ""

app = FastAPI()

# ── Request/Response models ───────────────────────────────────────────────────
class InvokeRequest(BaseModel):
    stream_mode: Literal["updates", "values"] = "values"  # values gives you final messages
    prompt: constr(min_length=3, strip_whitespace=True)
    thread_id: UUID

class InvokeResponse(BaseModel):
    content: str

# ── Assistant resolver ────────────────────────────────────────────────────────
async def _resolve_assistant_id() -> str:
    try:
        assistants = await client.assistants.search(graph_id="askdata_getinsights")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Assistant search failed: {e}")
    if not assistants:
        raise HTTPException(status_code=404, detail="No assistant found for graph_id='askdata_getinsights'.")
    aid = assistants[0].get("assistant_id")
    if not aid:
        raise HTTPException(status_code=502, detail="Assistant missing 'assistant_id'.")
    return aid


# Search all hosted graphs
@app.get("/askdata/search_assistant_id")
async def search():
    assistants = await client.assistants.search(graph_id= "askdata_getinsights")
    assistant_id = assistants[0]["assistant_id"] # Unique ID e.g 0468dc38-81bf-5b14-969d-81bd9f36e07d

    print(f"Assistant ID: '{assistant_id}'")
    return assistant_id

#create new thread/session
@app.get("/askdata/getsession")
async def create_thread():
    # Create a thread
    thread = await client.threads.create()
    print(f"ThreadId: '{thread['thread_id']}'")

    return thread['thread_id']

# ── POST endpoint (robust streaming parse) ────────────────────────────────────
@app.post("/askdata/getinsights", response_model=InvokeResponse)
async def invoke(req: InvokeRequest = Body(...)):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    assistant_id = await _resolve_assistant_id()
    input_message = HumanMessage(content=prompt)

    # Buffers
    last_ai_content: Optional[str] = None
    last_any_content: Optional[str] = None
    updates_buffer: list[str] = []
    summary_text: Optional[str] = None

    try:
        async for part in client.runs.stream(
            str(req.thread_id),
            assistant_id=assistant_id,
            input={"messages": [input_message]},
            stream_mode=req.stream_mode,
        ):
            # Support both StreamPart objects and (event, data) tuples
            if hasattr(part, "event"):
                event = part.event
                data = part.data
            else:
                event, data = part

            # Handle "values" (final structured messages)
            if event == "values":
                if isinstance(data, dict):
                    # messages: list of {type: 'ai'|'tool'|..., content: str, ...}
                    messages = data.get("messages", [])
                    for m in messages:
                        content = m.get("content")
                        if not content:
                            continue
                        last_any_content = content
                        if m.get("type") == "ai":
                            last_ai_content = content
                    # optional summary
                    if "summary" in data and isinstance(data["summary"], str):
                        summary_text = data["summary"]

            # Handle "updates" (incremental)
            elif event == "updates":
                # Different providers use different keys; capture anything reasonable
                # common shapes: {"delta": "..."} or {"content": "..."} or {"chunk": "..."}
                for k in ("delta", "content", "chunk", "text"):
                    v = data.get(k) if isinstance(data, dict) else None
                    if isinstance(v, str) and v:
                        updates_buffer.append(v)
                        break

            # Some SDKs emit final message under generic 'message' event as well
            elif event in ("message", "ai_message"):
                if isinstance(data, dict):
                    c = data.get("content")
                    if isinstance(c, str) and c:
                        last_any_content = c
                        if data.get("type") == "ai":
                            last_ai_content = c

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Streaming from assistant failed: {e}")

    # Decide what to return, in order of preference
    if last_ai_content:
        return InvokeResponse(content=last_ai_content)
    if last_any_content:
        return InvokeResponse(content=last_any_content)
    if updates_buffer:
        return InvokeResponse(content="".join(updates_buffer).strip())
    if summary_text:
        return InvokeResponse(content=summary_text)

    # Nothing usable
    raise HTTPException(status_code=502, detail="No content returned from assistant.")

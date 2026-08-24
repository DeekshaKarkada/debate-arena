"""FastAPI layer wrapping the debate graph.

Endpoints:
  POST /debates                      -> start a debate, returns thread_id immediately
  GET  /debates/{thread_id}/stream   -> SSE stream of each turn; pauses before judging
  POST /debates/{thread_id}/resume   -> approve (optionally with a note) and get the verdict
  GET  /debates/{thread_id}          -> current state (poll or inspect after the fact)

Human-in-the-loop: the graph is compiled with interrupt_before=["judge"], so
execution genuinely stops after the moderator decides to end the debate and
BEFORE the judge runs. Nothing happens until /resume is called against the
same thread_id — that's the real pause, not a simulated one.

Run with:  uvicorn app.main:app --reload
"""

import asyncio
import json
import uuid

from dotenv import load_dotenv

load_dotenv()  # must run before app.graph -> app.nodes reads LLM_PROVIDER / API keys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.graph import debate_app

app = FastAPI(title="Debate Arena")

# Streamlit will run on a different origin (localhost:8501 locally, a
# different domain once deployed) than this API — CORS must allow it,
# or the browser will silently block every request from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your deployed Streamlit URL before sharing publicly
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class StartDebateRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    max_turns: int = Field(default=4, ge=2, le=12)


class StartDebateResponse(BaseModel):
    thread_id: str


class ResumeDebateRequest(BaseModel):
    human_note: str | None = Field(
        default=None,
        max_length=500,
        description="Optional closing remark injected into the transcript before judging.",
    )


# ---------------------------------------------------------------------------
# Shared streaming helper
# ---------------------------------------------------------------------------

async def _stream_graph(config: dict):
    """Runs the graph forward from its current checkpoint and yields SSE
    events until it either finishes or hits the next interrupt.

    Used by both /stream (fresh run) and /resume (continuing after a human
    paused at the judge interrupt) — the underlying LangGraph call is
    identical (`stream(None, ...)` = "continue from the saved checkpoint").
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_graph():
        try:
            for event in debate_app.stream(None, config=config, stream_mode="values"):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel: this leg is done

    loop.run_in_executor(None, run_graph)

    last_message_count = 0  # tracks how many messages we've already forwarded

    while True:
        event = await queue.get()
        if event is None:
            break

        messages = event.get("messages", [])
        payload = {"status": event.get("status")}

        # stream_mode="values" re-emits the FULL state after every node,
        # including nodes (like the moderator) that don't add a message.
        # Only forward a "speaker" field when a genuinely new message has
        # appeared since the last event, so agent_b's turn isn't reported
        # twice just because the moderator ran right after it.
        if len(messages) > last_message_count:
            last = messages[-1]
            if hasattr(last, "name") and last.name:
                payload["speaker"] = last.name
                payload["content"] = last.content
            last_message_count = len(messages)

        if event.get("verdict"):
            payload["verdict"] = event["verdict"]

        yield f"data: {json.dumps(payload)}\n\n"

    # Distinguish "paused at the human-in-the-loop interrupt" from "actually finished"
    snapshot = debate_app.get_state(config)
    if snapshot.next:
        yield f"data: {json.dumps({'event': 'awaiting_human_review'})}\n\n"
    else:
        yield f"data: {json.dumps({'event': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/debates", response_model=StartDebateResponse)
def start_debate(req: StartDebateRequest):
    """Creates a new debate thread. Does NOT run it yet — the graph only
    actually executes once something reads from /stream."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": req.topic,
        "messages": [],
        "turn": 0,
        "max_turns": req.max_turns,
        "status": "ongoing",
        "verdict": None,
    }

    # Seed the checkpoint so GET /debates/{thread_id} works even before
    # anyone opens the stream.
    debate_app.update_state(config, initial_state)

    return StartDebateResponse(thread_id=thread_id)


@app.get("/debates/{thread_id}/stream")
async def stream_debate(thread_id: str):
    """Streams the debate turns. Will stop and emit `awaiting_human_review`
    once the moderator decides to end the debate — the judge does NOT run
    automatically. Call POST /resume to get the verdict."""
    config = {"configurable": {"thread_id": thread_id}}

    existing = debate_app.get_state(config)
    if not existing.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    return StreamingResponse(_stream_graph(config), media_type="text/event-stream")


@app.post("/debates/{thread_id}/resume")
async def resume_debate(thread_id: str, req: ResumeDebateRequest):
    """Resumes a debate paused at the judge interrupt. If human_note is
    provided, it's injected into the transcript as a labeled human message
    before judging, so the judge's prompt actually sees it."""
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = debate_app.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="This debate is not paused for review.")

    if req.human_note:
        note = HumanMessage(content=req.human_note, name="human_reviewer")
        debate_app.update_state(config, {"messages": [note]})

    return StreamingResponse(_stream_graph(config), media_type="text/event-stream")


@app.get("/debates/{thread_id}")
def get_debate(thread_id: str):
    """Returns current stored state — works even after a server restart,
    since state is persisted in SQLite."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = debate_app.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    values = snapshot.values
    transcript = [
        {"speaker": m.name, "content": m.content}
        for m in values.get("messages", [])
        if hasattr(m, "name") and m.name
    ]

    return {
        "thread_id": thread_id,
        "topic": values.get("topic"),
        "status": values.get("status"),
        "turn": values.get("turn"),
        "transcript": transcript,
        "verdict": values.get("verdict"),
        "awaiting_human_review": bool(snapshot.next),
    }
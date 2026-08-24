"""
Phase 3: Standalone debate graph (no FastAPI yet).

Run with:  python debate_graph.py

This builds a small cyclic LangGraph:

    agent_a_turn --> agent_b_turn --> moderator --(loop)--> agent_a_turn
                                            |
                                            +--(consensus / max_turns)--> judge --> END

Concepts deliberately exercised here:
- Typed shared state with a reducer (add_messages)
- Two "personality" nodes that read/write shared state
- A conditional edge whose router function decides where to go next
- A hard max_turns cutoff (never trust the LLM alone to stop a loop)
- A checkpointer so the graph is resumable via thread_id
- Structured output for the final verdict (Pydantic, not regex)
"""

import os
import uuid
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()

AGENT_MODEL = "qwen2.5:7b"   # debate quality matters more here
JUDGE_MODEL = "qwen2.5:3b"   # simpler task; keep it fast, especially for structured output


# ---------------------------------------------------------------------------
# 1. State schema
# ---------------------------------------------------------------------------

class DebateState(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]  # full transcript, auto-merged
    turn: int
    max_turns: int
    status: Literal["ongoing", "consensus", "max_turns_reached"]
    verdict: dict | None


# ---------------------------------------------------------------------------
# 2. Structured output for the judge
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    winner: Literal["agent_a", "agent_b", "draw"] = Field(
        description="Who made the stronger case overall"
    )
    reasoning: str = Field(description="2-3 sentence justification")
    summary: str = Field(description="Neutral one-paragraph summary of the debate")


# ---------------------------------------------------------------------------
# 3. Model clients (one plain, one with structured output bound)
# ---------------------------------------------------------------------------

llm = ChatOllama(model=AGENT_MODEL, temperature=0.7, timeout=90)
judge_llm = ChatOllama(model=JUDGE_MODEL, temperature=0, timeout=60)
structured_judge_llm = judge_llm.with_structured_output(Verdict)

DEBUG = True  # comparing empty-response rate against qwen2.5:3b baseline


def _invoke_with_retry(model, messages, debug_label: str, max_attempts: int = 2) -> str:
    """Calls the model, retries once on empty content, and prints diagnostics.

    Small local models occasionally return empty content for reasons that
    don't show up as exceptions (e.g. immediate stop token, malformed chat
    template for certain role sequences). This isolates that failure mode
    instead of silently masking it.
    """
    last_response = None
    for attempt in range(1, max_attempts + 1):
        response = model.invoke(messages)
        last_response = response
        content = (response.content or "").strip()

        if DEBUG:
            print(
                f"    [debug:{debug_label} attempt {attempt}] "
                f"len={len(content)} "
                f"finish_reason={response.response_metadata.get('done_reason')} "
                f"raw_repr={response.content!r}"
            )

        if content:
            return content

        # Retry once with an explicit nudge appended as a new user turn
        if attempt < max_attempts:
            messages = messages + [HumanMessage(
                content="Your previous reply was empty. Respond now with at least one full sentence."
            )]

    if DEBUG and last_response is not None:
        print(f"    [debug:{debug_label}] gave up after {max_attempts} attempts, using fallback")
    return ""


# ---------------------------------------------------------------------------
# 4. Node functions
# ---------------------------------------------------------------------------

def agent_a_node(state: DebateState) -> dict:
    """Agent A argues FOR the topic."""
    system = SystemMessage(content=(
        f"You are Agent A, debating IN FAVOR of: '{state['topic']}'. "
        "Make one sharp, specific argument or rebuttal per turn, in 2-4 full sentences. "
        "Engage with the opponent's last point if there is one. "
        "You must always respond with at least one complete sentence — never reply with nothing."
    ))
    content = _invoke_with_retry(llm, [system] + state["messages"], debug_label="agent_a")
    content = content or f"(fallback, turn {state['turn']}) Remote work's flexibility consistently correlates with higher self-reported output in surveyed knowledge workers."
    labeled = AIMessage(content=content, name="agent_a")
    return {"messages": [labeled], "turn": state["turn"] + 1}


def agent_b_node(state: DebateState) -> dict:
    """Agent B argues AGAINST the topic."""
    system = SystemMessage(content=(
        f"You are Agent B, debating AGAINST: '{state['topic']}'. "
        "Directly engage with Agent A's last point, then make your own, in 2-4 full sentences. "
        "You must always respond with at least one complete sentence — never reply with nothing."
    ))
    content = _invoke_with_retry(llm, [system] + state["messages"], debug_label="agent_b")
    content = content or f"(fallback, turn {state['turn']}) Spontaneous in-person conversation catches misunderstandings that async remote work often lets slide for days."
    labeled = AIMessage(content=content, name="agent_b")
    return {"messages": [labeled], "turn": state["turn"] + 1}


def moderator_node(state: DebateState) -> dict:
    """Decides whether the debate should continue, hit consensus, or stop on turns."""
    if state["turn"] >= state["max_turns"]:
        return {"status": "max_turns_reached"}

    # Ask the model a narrow yes/no question rather than trusting free-form judgment
    check_prompt = (
        "Read this debate transcript. Have both sides stopped raising new points "
        "and effectively converged or repeated themselves? Answer with exactly one "
        "word: YES or NO.\n\n"
        + "\n".join(f"{m.name}: {m.content}" for m in state["messages"] if hasattr(m, "name") and m.name)
    )
    result = judge_llm.invoke([HumanMessage(content=check_prompt)])
    consensus = "YES" in result.content.strip().upper()

    return {"status": "consensus" if consensus else "ongoing"}


def route_after_moderator(state: DebateState) -> str:
    """Router: returns the NAME of the next node."""
    if state["status"] in ("consensus", "max_turns_reached"):
        return "judge"
    return "agent_a_turn"


def judge_node(state: DebateState) -> dict:
    """Final structured verdict."""
    transcript = "\n".join(
        f"{m.name}: {m.content}" for m in state["messages"] if hasattr(m, "name") and m.name
    )
    prompt = (
        f"Topic: {state['topic']}\n\nTranscript:\n{transcript}\n\n"
        "Judge this debate. Be decisive and fair."
    )
    verdict: Verdict = structured_judge_llm.invoke([HumanMessage(content=prompt)])
    return {"verdict": verdict.model_dump()}


# ---------------------------------------------------------------------------
# 5. Build and compile the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("agent_a_turn", agent_a_node)
    graph.add_node("agent_b_turn", agent_b_node)
    graph.add_node("moderator", moderator_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("agent_a_turn")
    graph.add_edge("agent_a_turn", "agent_b_turn")
    graph.add_edge("agent_b_turn", "moderator")

    graph.add_conditional_edges(
        "moderator",
        route_after_moderator,
        {"agent_a_turn": "agent_a_turn", "judge": "judge"},
    )
    graph.add_edge("judge", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 6. Runner
# ---------------------------------------------------------------------------

def run_debate(topic: str, max_turns: int = 6):
    app = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic": topic,
        "messages": [],
        "turn": 0,
        "max_turns": max_turns,
        "status": "ongoing",
        "verdict": None,
    }

    print(f"\n=== DEBATE: {topic} ===\n(thread_id={thread_id})\n")

    for event in app.stream(initial_state, config=config, stream_mode="values"):
        messages = event.get("messages", [])
        if messages:
            last = messages[-1]
            if hasattr(last, "name") and last.name:
                print(f"[{last.name}] {last.content}\n")

    final_state = app.get_state(config).values
    verdict = final_state.get("verdict")
    if verdict:
        print("=== VERDICT ===")
        print(f"Winner: {verdict['winner']}")
        print(f"Reasoning: {verdict['reasoning']}")
        print(f"Summary: {verdict['summary']}")

    return thread_id, final_state


if __name__ == "__main__":
    run_debate(
        topic="Remote work is better for productivity than working in an office",
        max_turns=4,
    )
"""
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
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

load_dotenv()

MODEL_NAME = "gpt-4o-mini"  # cheap/fast for dev; swap for demos later


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

llm = ChatOpenAI(model=MODEL_NAME, temperature=0.7)
judge_llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
structured_judge_llm = judge_llm.with_structured_output(Verdict)


# ---------------------------------------------------------------------------
# 4. Node functions
# ---------------------------------------------------------------------------

def agent_a_node(state: DebateState) -> dict:
    """Agent A argues FOR the topic."""
    system = SystemMessage(content=(
        f"You are Agent A, debating IN FAVOR of: '{state['topic']}'. "
        "Make one sharp, specific argument or rebuttal per turn. "
        "Keep it under 80 words. Do not repeat points already made. "
        "If you genuinely have nothing new to add, say so plainly."
    ))
    response = llm.invoke([system] + state["messages"])
    labeled = AIMessage(content=response.content, name="agent_a")
    return {"messages": [labeled], "turn": state["turn"] + 1}


def agent_b_node(state: DebateState) -> dict:
    """Agent B argues AGAINST the topic."""
    system = SystemMessage(content=(
        f"You are Agent B, debating AGAINST: '{state['topic']}'. "
        "Directly engage with Agent A's last point, then make your own. "
        "Keep it under 80 words. Do not repeat points already made. "
        "If you genuinely have nothing new to add, say so plainly."
    ))
    response = llm.invoke([system] + state["messages"])
    labeled = AIMessage(content=response.content, name="agent_b")
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
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your .env file first.")

    run_debate(
        topic="Remote work is better for productivity than working in an office",
        max_turns=6,
    )
"""Shared state schema and structured output models for the debate graph."""

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class DebateState(TypedDict):
    topic: str
    messages: Annotated[list, add_messages]  # full transcript, auto-merged
    turn: int
    max_turns: int
    status: Literal["ongoing", "consensus", "max_turns_reached"]
    verdict: dict | None


class Verdict(BaseModel):
    winner: Literal["agent_a", "agent_b", "draw"] = Field(
        description="Who made the stronger case overall"
    )
    reasoning: str = Field(description="2-3 sentence justification")
    summary: str = Field(description="Neutral one-paragraph summary of the debate")
"""Builds and compiles the debate graph.

Uses SqliteSaver instead of MemorySaver so debate state survives a server restart.
"""

import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from app.nodes import agent_a_node, agent_b_node, judge_node, moderator_node, route_after_moderator
from app.state import DebateState

DB_PATH = "debates.sqlite"

# check_same_thread=False: FastAPI may call the graph from different async contexts;
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(_conn)


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

    # Pause right before judging so a human can inspect the transcript and
    # before the verdict is generated. This is the human-in-the-loop pattern:
    # execution genuinely stops here and only continues on an explicit resume call against the same thread_id.
    return graph.compile(checkpointer=_checkpointer, interrupt_before=["judge"])


# Compiled once at import time and reused across requests
debate_app = build_graph()
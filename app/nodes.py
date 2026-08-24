"""Model clients and node functions for the debate graph.

Provider is controlled by LLM_PROVIDER in .env: "ollama" | "openai" | "anthropic".
Swap providers by changing that one value — nothing else in this file needs
to change, since LangChain abstracts the provider behind the same ChatModel
interface (this is the payoff of using LangChain's model wrappers).
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.state import DebateState, Verdict

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
DEBUG = os.getenv("DEBATE_DEBUG", "false").lower() == "true"


def _build_clients():
    """Returns (agent_llm, judge_llm) for the configured provider."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        agent_model = os.getenv("OPENAI_AGENT_MODEL", "gpt-4o-mini")
        judge_model = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        return (
            ChatOpenAI(model=agent_model, temperature=0.7, timeout=60),
            ChatOpenAI(model=judge_model, temperature=0, timeout=60),
        )

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        agent_model = os.getenv("ANTHROPIC_AGENT_MODEL", "claude-haiku-4-5-20251001")
        judge_model = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-haiku-4-5-20251001")
        return (
            ChatAnthropic(model=agent_model, temperature=0.7, timeout=60),
            ChatAnthropic(model=judge_model, temperature=0, timeout=60),
        )

    # default: local Ollama
    from langchain_ollama import ChatOllama
    agent_model = os.getenv("OLLAMA_AGENT_MODEL", "qwen2.5:7b")
    judge_model = os.getenv("OLLAMA_JUDGE_MODEL", "qwen2.5:3b")
    return (
        ChatOllama(model=agent_model, temperature=0.7, timeout=90),
        ChatOllama(model=judge_model, temperature=0, timeout=60),
    )


llm, judge_llm = _build_clients()
structured_judge_llm = judge_llm.with_structured_output(Verdict)


def _invoke_with_retry(model, messages, debug_label: str, max_attempts: int = 2) -> str:
    """Calls the model, retries once on empty content, and optionally logs diagnostics.

    Small/local models occasionally return empty content for reasons that
    don't show up as exceptions (e.g. immediate stop token). This isolates
    that failure mode instead of silently masking it.
    """
    for attempt in range(1, max_attempts + 1):
        response = model.invoke(messages)
        content = (response.content or "").strip()

        if DEBUG:
            print(
                f"    [debug:{debug_label} attempt {attempt}] "
                f"len={len(content)} "
                f"finish_reason={response.response_metadata.get('done_reason')}"
            )

        if content:
            return content

        if attempt < max_attempts:
            messages = messages + [HumanMessage(
                content="Your previous reply was empty. Respond now with at least one full sentence."
            )]

    return ""


def agent_a_node(state: DebateState) -> dict:
    """Agent A argues FOR the topic."""
    round_num = (state["turn"] // 2) + 1
    system = SystemMessage(content=(
        f"You are Agent A, debating IN FAVOR of: '{state['topic']}'. "
        f"This is round {round_num} of the debate. "
        "Make one sharp, specific argument or rebuttal per turn, in 2-4 full sentences. "
        "Engage with the opponent's last point if there is one. "
        "Do NOT repeat or rephrase a point you have already made earlier in this debate — "
        "review your own prior turns above and bring a genuinely new angle, example, or rebuttal each time. "
        "You must always respond with at least one complete sentence — never reply with nothing."
    ))
    content = _invoke_with_retry(llm, [system] + state["messages"], debug_label="agent_a")
    content = content or (
        f"(fallback, turn {state['turn']}) Remote work's flexibility consistently correlates "
        "with higher self-reported output in surveyed knowledge workers."
    )
    labeled = AIMessage(content=content, name="agent_a")
    return {"messages": [labeled], "turn": state["turn"] + 1}


def agent_b_node(state: DebateState) -> dict:
    """Agent B argues AGAINST the topic."""
    round_num = (state["turn"] // 2) + 1
    system = SystemMessage(content=(
        f"You are Agent B, debating AGAINST: '{state['topic']}'. "
        f"This is round {round_num} of the debate. "
        "Directly engage with Agent A's last point, then make your own, in 2-4 full sentences. "
        "Do NOT repeat or rephrase a point you have already made earlier in this debate — "
        "review your own prior turns above and bring a genuinely new angle, example, or rebuttal each time. "
        "You must always respond with at least one complete sentence — never reply with nothing."
    ))
    content = _invoke_with_retry(llm, [system] + state["messages"], debug_label="agent_b")
    content = content or (
        f"(fallback, turn {state['turn']}) Spontaneous in-person conversation catches "
        "misunderstandings that async remote work often lets slide for days."
    )
    labeled = AIMessage(content=content, name="agent_b")
    return {"messages": [labeled], "turn": state["turn"] + 1}


def moderator_node(state: DebateState) -> dict:
    """Decides whether the debate should continue, hit consensus, or stop on turns."""
    if state["turn"] >= state["max_turns"]:
        return {"status": "max_turns_reached"}

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
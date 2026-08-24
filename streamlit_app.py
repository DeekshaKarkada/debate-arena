"""Streamlit frontend for the LangGraph debate arena.

Talks to the FastAPI backend over HTTP/SSE — this file has no LangGraph or
model logic of its own, it's purely a UI over the API you already built.

Run locally with:
    streamlit run streamlit_app.py

Configure the backend URL via Streamlit secrets (.streamlit/secrets.toml):
    API_URL = "http://127.0.0.1:8000"
or an environment variable API_URL, falling back to localhost for local dev.
"""

import json
import os

import requests
import streamlit as st

def _get_api_url() -> str:
    """Reads API_URL from Streamlit secrets if available, falling back to
    an environment variable, then localhost for local dev.

    st.secrets raises (rather than returning a default) if no
    secrets.toml file exists at all locally — .get() doesn't protect
    against that, so this needs an explicit try/except.
    """
    try:
        return st.secrets["API_URL"]
    except Exception:
        # Streamlit raises StreamlitSecretNotFoundError (not just KeyError)
        # when no secrets.toml exists anywhere at all — broad catch here
        # is intentional: any failure to read secrets should just fall
        # back to the env var / localhost default, not crash the app.
        return os.getenv("API_URL", "http://127.0.0.1:8000")


API_URL = _get_api_url()

st.set_page_config(page_title="Debate Arena", page_icon="🗣️", layout="centered")

SPEAKER_LABELS = {
    "agent_a": "🟦 Agent A — For",
    "agent_b": "🟥 Agent B — Against",
    "human_reviewer": "🧑 Human note",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "thread_id": None,
        "topic": None,
        "transcript": [],       # list of (speaker, content)
        "verdict": None,
        "awaiting_review": False,
        "error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_state():
    st.session_state.thread_id = None
    st.session_state.topic = None
    st.session_state.transcript = []
    st.session_state.verdict = None
    st.session_state.awaiting_review = False
    st.session_state.error = None


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def render_message(speaker: str, content: str):
    label = SPEAKER_LABELS.get(speaker, speaker)
    st.markdown(f"**{label}**")
    st.write(content)
    st.write("")


def render_verdict(verdict: dict):
    st.subheader("🏆 Verdict")
    winner_label = SPEAKER_LABELS.get(verdict["winner"], verdict["winner"])
    st.markdown(f"**Winner:** {winner_label}")
    st.markdown(f"**Reasoning:** {verdict['reasoning']}")
    st.markdown(f"**Summary:** {verdict['summary']}")


def stream_and_collect(url: str, method: str = "GET", json_body: dict | None = None):
    """Consumes an SSE stream from the backend, updating session_state and
    rendering each new message live as it arrives."""
    try:
        with requests.request(method, url, json=json_body, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                payload = json.loads(raw_line[len("data:"):].strip())

                if "speaker" in payload:
                    entry = (payload["speaker"], payload["content"])
                    # The backend re-emits the same last message after the
                    # moderator step (a harmless artifact of stream_mode=
                    # "values") — skip exact consecutive duplicates.
                    if not st.session_state.transcript or st.session_state.transcript[-1] != entry:
                        st.session_state.transcript.append(entry)
                        render_message(*entry)

                if "verdict" in payload:
                    st.session_state.verdict = payload["verdict"]

                if payload.get("event") == "awaiting_human_review":
                    st.session_state.awaiting_review = True
                if payload.get("event") == "done":
                    st.session_state.awaiting_review = False

    except requests.exceptions.RequestException as exc:
        st.session_state.error = f"Couldn't reach the debate API: {exc}"


# ---------------------------------------------------------------------------
# Sidebar — start a new debate
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("New debate")
    topic_input = st.text_area(
        "Topic",
        value="Remote work is better for productivity than working in an office",
        height=80,
    )
    max_turns = st.slider("Max turns", min_value=2, max_value=12, value=4, step=2)

    if st.button("Start debate", type="primary", use_container_width=True):
        _reset_state()
        try:
            resp = requests.post(
                f"{API_URL}/debates",
                json={"topic": topic_input, "max_turns": max_turns},
                timeout=30,
            )
            resp.raise_for_status()
            st.session_state.thread_id = resp.json()["thread_id"]
            st.session_state.topic = topic_input
        except requests.exceptions.RequestException as exc:
            st.session_state.error = f"Couldn't start a debate: {exc}"

    st.divider()
    st.caption(
        "Built with LangGraph (cyclic multi-agent graph, checkpointed state, "
        "human-in-the-loop interrupt) behind a FastAPI + SSE backend."
    )


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.title("🗣️ Debate Arena")

if st.session_state.error:
    st.error(st.session_state.error)

if not st.session_state.thread_id:
    st.info("Set a topic in the sidebar and click **Start debate**.")
    st.stop()

st.markdown(f"**Topic:** {st.session_state.topic}")
st.divider()

# Replay whatever's already in session_state (covers reruns after a resume click)
for speaker, content in st.session_state.transcript:
    render_message(speaker, content)

# Kick off the initial stream exactly once per new thread
if not st.session_state.transcript and not st.session_state.verdict and not st.session_state.error:
    with st.spinner("Debate in progress..."):
        stream_and_collect(f"{API_URL}/debates/{st.session_state.thread_id}/stream")
    st.rerun()

# Paused for human review
if st.session_state.awaiting_review and not st.session_state.verdict:
    st.info("⏸️ Debate paused before judging — this is the human-in-the-loop checkpoint.")
    note = st.text_area(
        "Optional note for the judge (e.g. steer the criteria before the verdict)",
        key="human_note_input",
    )
    if st.button("Resume & get verdict", type="primary"):
        with st.spinner("Judging..."):
            stream_and_collect(
                f"{API_URL}/debates/{st.session_state.thread_id}/resume",
                method="POST",
                json_body={"human_note": note or None},
            )
        st.rerun()

# Final verdict
if st.session_state.verdict:
    st.divider()
    render_verdict(st.session_state.verdict)
    if st.button("Start a new debate"):
        _reset_state()
        st.rerun()
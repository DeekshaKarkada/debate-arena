# Debate Arena 

Two LLM agents debate opposite sides of a topic. A moderator agent decides when
the debate has run its course. Execution then **pauses for human review**
before a judge agent delivers a structured verdict — a person can add a note
that visibly shapes the final ruling before choosing to resume.

**Live demo:** [https://debateloop.streamlit.app/]
**Backend API:** [https://debate-arena-f8q9.onrender.com]

![demo img](image.png)

---

## Why this exists

This is a portfolio project built specifically to exercise the concepts that
differentiate [LangGraph](https://langchain-ai.github.io/langgraph/) from a
simple prompt-chaining script:

- **Cyclic control flow** — the two agents loop back and forth through a
  moderator-controlled conditional edge, not a fixed linear chain.
- **Shared, typed state with reducers** — the full transcript accumulates
  correctly across nodes via `add_messages`.
- **Durable, checkpointed execution** — state is persisted to SQLite via
  `SqliteSaver`, so a debate survives a server restart. `GET /debates/{id}`
  works identically before and after redeploying.
- **Genuine human-in-the-loop** — the graph is compiled with
  `interrupt_before=["judge"]`. Execution actually halts; nothing continues
  until an explicit `POST /resume` call against the same `thread_id`.
- **Streaming architecture** — LangGraph's `.stream()` is bridged into
  Server-Sent Events over FastAPI, so a client sees each turn as it's
  generated instead of waiting for the whole debate to finish.
- **Provider-agnostic model config** — `LLM_PROVIDER` swaps between local
  Ollama models and paid OpenAI/Anthropic APIs with no code changes, only an
  env var — useful for free local development and cheap final demos.


## Tech stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — agent
  orchestration (state, nodes, conditional edges, checkpointing, interrupts)
- **FastAPI** — REST + SSE streaming API
- **Streamlit** — frontend
- **SQLite** (via `langgraph-checkpoint-sqlite`) — persistent state
- **Ollama / OpenAI / Anthropic** — pluggable model backends via LangChain's
  chat model interface

## API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/debates` | Start a debate. Returns `thread_id` immediately. |
| `GET` | `/debates/{thread_id}/stream` | SSE stream of turns. Pauses before judging. |
| `POST` | `/debates/{thread_id}/resume` | Optionally inject a human note, then get the verdict. |
| `GET` | `/debates/{thread_id}` | Current stored state (works after restart). |

Interactive docs available at `/docs` once the backend is running (FastAPI's
built-in Swagger UI).

## Running locally

### 1. Clone and install
```bash
git clone <your-repo-url>
cd debate-arena
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure a model provider
```bash
cp .env.example .env
```
Default is `LLM_PROVIDER=ollama` (free, local, no key needed) — just make
sure [Ollama](https://ollama.com) is running and you've pulled a model:
```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5:3b
```
To use a paid API instead, set `LLM_PROVIDER=openai` or `anthropic` in
`.env` and add the matching key.

### 3. Run the backend
```bash
uvicorn app.main:app --reload
```

### 4. Run the frontend (separate terminal)
```bash
streamlit run streamlit_app.py
```
Opens at `http://localhost:8501`.

### 5. (Optional) CLI test client, no frontend needed
```bash
python test_client.py
```

## Known limitations

- **Local model coherence**: with small local models (e.g. `qwen2.5:3b`) on
  CPU-only inference, agents occasionally repeat earlier arguments or
  produce empty completions despite explicit anti-repetition prompting.
  A retry-with-nudge layer in `app/nodes.py` handles the empty-completion
  case; the repetition issue is substantially reduced with `qwen2.5:7b` and
  resolved when using a paid API model (`gpt-4o-mini` / Claude Haiku).
- **Consensus detection**: the moderator currently ends debates almost
  entirely via the `max_turns` cutoff rather than detecting genuine
  consensus — worth tuning the moderator prompt further with more debate
  data.
- **Render free-tier cold starts**: the deployed backend may take 30–60s to
  respond on its first request after a period of inactivity.

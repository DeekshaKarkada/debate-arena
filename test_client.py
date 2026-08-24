"""Manual test for the FastAPI + SSE debate endpoint, including the
human-in-the-loop resume step.

Run the server first:  uvicorn app.main:app --reload
Then, in another terminal:  python test_client.py
"""

import json

import httpx

BASE_URL = "http://127.0.0.1:8000"


def print_stream(response: httpx.Response):
    """Prints turns as they arrive and returns True if the stream ended
    awaiting human review (vs. fully done)."""
    awaiting_review = False
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = json.loads(line.removeprefix("data:").strip())

        if "speaker" in payload:
            print(f"[{payload['speaker']}] {payload['content']}\n")
        if "verdict" in payload:
            v = payload["verdict"]
            print("=== VERDICT ===")
            print(f"Winner: {v['winner']}")
            print(f"Reasoning: {v['reasoning']}")
            print(f"Summary: {v['summary']}\n")
        if payload.get("event") == "awaiting_human_review":
            awaiting_review = True
        if payload.get("event") == "done":
            print("[stream finished]\n")

    return awaiting_review


def main():
    # 1. Start a debate
    resp = httpx.post(
        f"{BASE_URL}/debates",
        json={"topic": "Cats are better pets than dogs", "max_turns": 4},
    )
    resp.raise_for_status()
    thread_id = resp.json()["thread_id"]
    print(f"Started debate: {thread_id}\n")

    # 2. Stream it — this will stop BEFORE the judge runs
    with httpx.stream("GET", f"{BASE_URL}/debates/{thread_id}/stream", timeout=180) as stream:
        awaiting_review = print_stream(stream)

    if not awaiting_review:
        print("Debate finished without pausing for review (unexpected) — check interrupt_before config.")
        return

    print(">>> Debate is paused, waiting for human review.")
    print(">>> Injecting a human note and resuming...\n")

    # 3. Resume — with an optional human note the judge will see
    with httpx.stream(
        "POST",
        f"{BASE_URL}/debates/{thread_id}/resume",
        json={"human_note": "Consider which pet is better for small children specifically."},
        timeout=180,
    ) as stream:
        print_stream(stream)

    # 4. Fetch stored state (proves persistence + the human note landed)
    final = httpx.get(f"{BASE_URL}/debates/{thread_id}").json()
    print(f"Stored status: {final['status']}, turns: {final['turn']}, "
          f"awaiting_human_review: {final['awaiting_human_review']}")


if __name__ == "__main__":
    main()
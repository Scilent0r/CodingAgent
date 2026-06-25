"""
Renders a run's transcript.jsonl as a readable timeline.

Usage:
    python view_transcript.py runs/20260621-215344
"""
import argparse
import json
from pathlib import Path

DIVIDER = "-" * 70


def render_entry(e: dict):
    if "git_setup" in e:
        print(f"\n=== Git setup ===")
        print(e["git_setup"])
        print(DIVIDER)
        return

    if "fatal_llm_error" in e:
        print(f"\n=== Iteration {e['iteration']} : FATAL LLM ERROR ===")
        print(e["fatal_llm_error"])
        return

    if "fatal_error" in e:
        print(f"\n=== Iteration {e['iteration']} : ABORTED ===")
        print(e["fatal_error"])
        return

    print(f"\n=== Iteration {e['iteration']} : {e.get('action', '?').upper()} ===")
    if e.get("thought"):
        print(f"Thought: {e['thought']}")
    if e.get("path"):
        print(f"Path: {e['path']}")

    action = e.get("action")
    if action == "edit_file" and "old_str" in e:
        print("--- old ---")
        print(_truncate(e.get("old_str", "")))
        print("--- new ---")
        print(_truncate(e.get("new_str", "")))
    elif action == "write_file" and "content_len" in e:
        print(f"(wrote {e['content_len']} chars)")
    elif action == "update_plan" and "plan" in e:
        print("Plan:")
        print(e["plan"])
    elif action == "update_memory" and "memory_note" in e:
        print(f"Saved to project memory: {e['memory_note']}")
    elif action == "ask_user" and "question" in e:
        print(f"Question to user: {e['question']}")
        print(f"User answered: {e.get('user_answer', '')}")
    elif action == "web_research" and "research_confidence" in e:
        print(f"Research confidence: {e['research_confidence']}")
        print(f"Sources consulted: {e.get('research_sources', [])}")

    if "critic_approved" in e:
        verdict = "APPROVED" if e["critic_approved"] else "REJECTED"
        print(f"Critic verdict: {verdict} — {e.get('critic_reason', '')}")
        if e.get("critic_concerns"):
            print(f"Critic concerns: {e['critic_concerns']}")

    obs = e.get("observation", "")
    if obs:
        print("Observation:")
        print(_truncate(obs, 500))

    print(DIVIDER)


def _truncate(text: str, limit: int = 300) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [{len(text) - limit} more chars]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Path to a runs/<timestamp> directory")
    args = parser.parse_args()

    transcript_path = Path(args.run_dir) / "transcript.jsonl"
    if not transcript_path.exists():
        print(f"No transcript found at {transcript_path}")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            render_entry(json.loads(line))


if __name__ == "__main__":
    main()

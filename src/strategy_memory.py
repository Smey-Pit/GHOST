"""
src/strategy_memory.py

Persistent strategy memory for GHOST self-improving agent
(GHOST_self_improving.md, Component C: Persistent Memory with Retrieval).

Storage:
  results/strategy_memory.json — indexed by content profile
  results/experience_log.jsonl — raw experience log (append-only,
      Component A)

Paths are anchored to the repo root via this file's own location, not
CWD -- so behaviour doesn't change depending on whether a caller runs
`python src/convergence.py` from the repo root or `python
convergence.py` from inside src/ (both invocation styles exist across
this project's scripts and docs).

The strategy_memory.json structure:
{
  "version": 1,
  "n_documents_processed": 47,
  "principles": [
    {
      "id": "uuid",
      "content_type": "multi_sentence",
      "n_chars_range": [80, 200],
      "has_numbers": false,
      "complexity": "high",
      "principle": "For multi-sentence high-complexity text...",
      "supporting_evidence": 3,  # times this worked
      "confidence": "medium",    # low/medium/high
      "first_seen": "2026-08-11T...",
      "last_updated": "2026-08-11T...",
    },
    ...
  ]
}
"""

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from ghost_tools_structural import tool_analyse_structure  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parent.parent
MEMORY_PATH = _REPO_ROOT / "results" / "strategy_memory.json"
EXPERIENCE_LOG_PATH = _REPO_ROOT / "results" / "experience_log.jsonl"


def load_memory() -> dict:
    """Load strategy memory from disk. Returns empty if none."""
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH) as f:
            return json.load(f)
    return {
        "version": 1,
        "n_documents_processed": 0,
        "principles": []
    }


def save_memory(memory: dict):
    """Atomically save memory to disk."""
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MEMORY_PATH.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(memory, f, indent=2)
    tmp.replace(MEMORY_PATH)


def append_experience(experience: dict):
    """Append one experience to the log. Never corrupts (JSONL, not JSON)."""
    EXPERIENCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIENCE_LOG_PATH, 'a') as f:
        f.write(json.dumps(experience) + '\n')


def retrieve_relevant_principles(
    text: str,
    memory: dict,
    top_k: int = 5
) -> list:
    """
    Retrieve principles relevant to this text.

    Matching priority:
    1. Exact content_type match
    2. Similar n_chars_range (within 50%)
    3. Same has_numbers flag
    4. Higher confidence first

    Returns list of principle strings, most relevant first.
    """
    profile = tool_analyse_structure(text)
    principles = memory.get("principles", [])

    def relevance_score(p: dict) -> float:
        score = 0.0
        if p.get("content_type") == profile["content_type"]:
            score += 3.0
        lo, hi = p.get("n_chars_range", [0, 9999])
        if lo <= profile["n_chars"] <= hi:
            score += 2.0
        elif abs(profile["n_chars"] - (lo + hi) / 2) < (hi - lo):
            score += 1.0
        if p.get("has_numbers") == profile["has_numbers"]:
            score += 1.0
        confidence_bonus = {"high": 1.0, "medium": 0.5, "low": 0.0}
        score += confidence_bonus.get(p.get("confidence", "low"), 0.0)
        return score

    scored = sorted(principles, key=relevance_score, reverse=True)
    return [p["principle"] for p in scored[:top_k] if relevance_score(p) > 0]


def add_principle(
    memory: dict,
    principle: str,
    content_profile: dict,
    bidi_config: str,
    vs_payload: str,
    n_iterations: int,
    adversary_model: str,
) -> dict:
    """
    Add a new principle or strengthen an existing one.

    If a principle with matching profile already exists, increment its
    supporting_evidence counter and upgrade confidence if warranted.
    Otherwise create a new principle entry.
    """
    n_chars = content_profile.get("n_chars", 0)

    for existing in memory["principles"]:
        if (existing["content_type"] == content_profile.get("content_type")
                and abs(existing["n_chars_range"][0] - n_chars) < 50):
            existing["supporting_evidence"] += 1
            existing["last_updated"] = datetime.now().isoformat()
            ev = existing["supporting_evidence"]
            existing["confidence"] = (
                "high" if ev >= 5 else
                "medium" if ev >= 2 else
                "low"
            )
            return memory

    lo = max(0, n_chars - 30)
    hi = n_chars + 30
    memory["principles"].append({
        "id": str(uuid.uuid4())[:8],
        "content_type": content_profile.get("content_type"),
        "n_chars_range": [lo, hi],
        "has_numbers": content_profile.get("has_numbers", False),
        "complexity": content_profile.get("complexity", "medium"),
        "principle": principle,
        "bidi_config": bidi_config,
        "vs_payload": vs_payload,
        "n_iterations": n_iterations,
        "adversary_model": adversary_model,
        "supporting_evidence": 1,
        "confidence": "low",
        "first_seen": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
    })
    return memory


def format_principles_for_prompt(principles: list) -> str:
    """
    Format retrieved principles for injection into the agent prompt.
    Returns empty string if no principles (so callers can skip
    appending anything to the system prompt).
    """
    if not principles:
        return ""

    lines = [
        "LEARNED STRATEGIES FROM PREVIOUS DOCUMENTS:",
        "Apply these before experimenting with new configs.",
        "",
    ]
    for i, p in enumerate(principles, 1):
        lines.append(f"{i}. {p}")
    lines.append("")
    return "\n".join(lines)

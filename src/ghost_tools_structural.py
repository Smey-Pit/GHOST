"""
src/ghost_tools_structural.py

Content-profile analysis for GHOST-Agent's strategy memory
(GHOST_self_improving.md, Component A/C: experience capture and
retrieval both need a content profile to key on).

Deliberately simple heuristics, not a learned classifier -- the profile
only needs to be specific enough to group "similar" targets for
retrieve_relevant_principles() in strategy_memory.py, not to be a
general-purpose text classifier.
"""

import re


def tool_analyse_structure(text: str) -> dict:
    """
    Produce a content profile for `text`.

    Returns:
        dict with keys:
          content_type: one of "numeric", "alphanumeric_id",
              "single_sentence", "multi_sentence", "long_text"
          n_chars: int
          n_sentences: int — count of .!? terminators (min 1)
          has_numbers: bool
          complexity: "low"/"medium"/"high" — heuristic on length and
              character diversity
    """
    n_chars = len(text)
    has_numbers = bool(re.search(r"\d", text))
    has_letters = bool(re.search(r"[a-zA-Z]", text))
    n_sentences = max(1, len(re.findall(r"[.!?]+", text)))
    n_words = len(text.split())

    if has_numbers and not has_letters:
        content_type = "numeric"
    elif has_numbers and has_letters and n_words <= 3:
        content_type = "alphanumeric_id"
    elif n_sentences >= 2 or n_words > 40:
        content_type = "multi_sentence" if n_chars <= 400 else "long_text"
    elif n_words > 1:
        content_type = "single_sentence"
    else:
        content_type = "alphanumeric_id"

    distinct_chars = len(set(text.lower()))
    if n_chars < 20 and distinct_chars < 12:
        complexity = "low"
    elif n_chars < 200:
        complexity = "medium"
    else:
        complexity = "high"

    return {
        "content_type": content_type,
        "n_chars": n_chars,
        "n_sentences": n_sentences,
        "has_numbers": has_numbers,
        "complexity": complexity,
    }

"""
src/reflection.py

Reflection memory for the GHOST self-improving agent (Task 3 of
GHOST_AGENT_TASKS.md).

Stores the history of encoding attempts and their outcomes, and formats
this history into a structured reflection the agent uses to reason
about what to try next. No API calls. Pure data management and prompt
formatting.

DEVIATION FROM THE ORIGINAL TASK SPEC: the spec's Attempt/reflection
design assumes a single adversary_response per attempt. Task 2 already
replaced that with an ensemble consensus vote (src/ensemble.py's
run_ensemble_query -> {defended, n_valid, n_failed, per_member}), so
Attempt stores that ensemble_result instead of a lone response --
matching what Task 5's agent loop will actually have to feed in.

The reflection prompt answers three questions:
  1. What did I try?
  2. What happened -- against EACH ensemble member, not just "the model"?
  3. What should I infer?
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attempt:
    """
    A single encoding attempt and its ensemble outcome.
    """
    iteration: int
    bidi_config: str           # config string used
    vs_payload: str            # payload string used
    encoding_description: str  # agent's own description
    encoded_text: str          # the actual encoded string
    rendered: str              # what human sees
    stored: str                # what model tokenizes
    hamming_dist: int          # stored vs target distance
    ensemble_result: dict      # run_ensemble_query() output:
                                # {defended, n_valid, n_failed, per_member}
    agent_reasoning: str       # agent's reflection on this

    @property
    def extraction_defeated(self) -> bool:
        """Did this attempt cross the ensemble's consensus threshold?"""
        return self.ensemble_result["defended"]

    @property
    def n_failed(self) -> int:
        return self.ensemble_result["n_failed"]

    @property
    def n_valid(self) -> int:
        return self.ensemble_result["n_valid"]

    @property
    def members_that_extracted(self) -> List[str]:
        """Names of valid members that still recovered the target."""
        return [
            m["name"] for m in self.ensemble_result["per_member"]
            if m["valid"] and m["extracted"]
        ]


@dataclass
class ReflectionMemory:
    """
    Accumulates attempt history for one document field.
    """
    target: str              # original field value
    field_name: str          # e.g. "account_number"
    attempts: List[Attempt] = field(default_factory=list)

    def add_attempt(self, attempt: Attempt):
        self.attempts.append(attempt)

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def best_attempt(self) -> Optional[Attempt]:
        """
        Attempt that defeated the most ensemble members, breaking ties
        by Hamming distance. n_failed is the more direct signal (it's
        what the agent is actually optimising) -- Hamming distance is
        only a proxy for it.
        """
        if not self.attempts:
            return None
        return max(
            self.attempts, key=lambda a: (a.n_failed, a.hamming_dist)
        )

    @property
    def succeeded(self) -> bool:
        """Has any attempt crossed the ensemble's consensus threshold?"""
        return any(a.extraction_defeated for a in self.attempts)

    def format_for_reflection(self) -> str:
        """
        Format attempt history for the agent's reflection prompt.
        Returns a structured string summarising what was tried, the
        per-member breakdown, and what outcomes were.
        """
        if not self.attempts:
            return "No attempts yet."

        lines = [
            f"TARGET: '{self.target}' (length {len(self.target)})",
            f"FIELD: {self.field_name}",
            f"ATTEMPTS SO FAR: {self.n_attempts}",
            "",
        ]

        for att in self.attempts:
            status = (
                "SUCCESS (ensemble consensus defeated)"
                if att.extraction_defeated
                else "FAILED (consensus threshold not reached)"
            )
            member_lines = [
                f"    - {m['name']}: "
                f"{'excluded (failed clean-floor check)' if not m['valid'] else ('REFUSED' if m['refusal'] else ('extracted correctly' if m['extracted'] else 'defeated'))}"
                for m in att.ensemble_result["per_member"]
            ]

            lines.extend([
                f"--- Attempt {att.iteration} ---",
                f"Configuration: {att.bidi_config}",
                f"VS payload: '{att.vs_payload}'",
                f"Stored sequence: '{att.stored}'",
                f"Hamming distance: {att.hamming_dist}",
                f"Ensemble result: {att.n_failed}/{att.n_valid} valid "
                f"members defeated",
                "Per-member breakdown:",
                *member_lines,
                f"Outcome: {status}",
                f"Your reasoning then: {att.agent_reasoning}",
                "",
            ])

        # Summary insights
        failed = [a for a in self.attempts if not a.extraction_defeated]
        succeeded = [a for a in self.attempts if a.extraction_defeated]

        if failed:
            best_n_failed = max(a.n_failed for a in failed)
            lines.append(
                f"INSIGHT: Best result before success: {best_n_failed} "
                f"ensemble members defeated"
            )
            # Which specific members keep getting it right?
            still_extracting = {}
            for a in failed:
                for name in a.members_that_extracted:
                    still_extracting[name] = still_extracting.get(name, 0) + 1
            if still_extracting:
                repeat_offenders = sorted(
                    still_extracting, key=still_extracting.get, reverse=True
                )
                lines.append(
                    "INSIGHT: These members keep extracting correctly -- "
                    f"{', '.join(repeat_offenders)}. Their tokenizers may "
                    "share a weakness your current bidi/VS combination "
                    "doesn't exploit; try a different config or stronger "
                    "injection rather than repeating the same approach."
                )

        if succeeded:
            lines.append(
                f"NOTE: {len(succeeded)} attempt(s) already crossed the "
                f"consensus threshold. Returning best one."
            )

        return "\n".join(lines)

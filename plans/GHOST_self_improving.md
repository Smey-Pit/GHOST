# Self-Improving Agent: Definition and Codebase Audit
# Read this document, audit the existing codebase,
# then report what is and is not implemented.

---

## PART 1: WHAT SELF-IMPROVING MEANS

### The Core Distinction

There are three different things people call "improvement"
in agentic systems. Only one of them is genuine
self-improvement. You need to identify which one the
current codebase is doing.

---

### Level 1: Output Refinement (NOT self-improving)

The agent produces better output within a single session.
The agent itself does not change.

```
Session 1, Document 1:
  Attempt 1: full_rtl → adversary succeeds
  Attempt 2: rli_split:4 → adversary succeeds  
  Attempt 3: combine + heavy VS → adversary fails ✓
  Session ends.

Session 1, Document 2:
  Attempt 1: full_rtl → adversary succeeds  ← starts over
  Attempt 2: ...
```

The agent learned nothing from Document 1 that helped
Document 2. It is doing iterative output refinement,
not self-improvement.

This is Reflexion (Shinn et al., 2023) and Self-Refine
(Madaan et al., 2023). Respected work. But NOT what the
self-improving agent literature (2024-2026) refers to.

---

### Level 2: In-Session Accumulation (partial)

The agent carries experience forward WITHIN a session but
resets between sessions.

```
Session 1:
  Document 1: 4 iterations, learns config A works
  Document 2: tries config A first → 2 iterations
  Document 3: tries config A → 1 iteration
  Session ends.

Session 2 (restart):
  Document 4: starts fresh again ← resets
```

Better than Level 1, but still not genuine self-improvement.
The knowledge does not persist.

---

### Level 3: Cross-Session Self-Improvement (the real thing)

The agent accumulates experience that PERSISTS across
sessions in external memory. When it restarts, it loads
what it learned and performs better from document 1.

```
Session 1 (Day 1):
  Documents 1-50: agent explores, learns strategies
  Saves to strategy_memory.json

Session 2 (Day 2):
  Loads strategy_memory.json
  Document 51: uses learned strategies → converges fast
  Documents 51-150: continues learning
  Saves to strategy_memory.json

Session 3 (Day 3):
  Loads updated strategy_memory.json
  Document 151: converges in 1 iteration (well-learned)
```

The empirical signature of genuine self-improvement:
ITERATIONS TO CONVERGENCE DECREASES AS DOCUMENTS INCREASE.
This must be measurable and plotted. If you cannot show
this curve, you cannot claim self-improvement.

---

### The Three Required Components

For Level 3 to exist, the system must have all three:

```
COMPONENT A: Experience Capture
  After each encoding attempt (success or failure),
  the agent records WHAT it tried and WHAT HAPPENED.
  Not just the result — the WHY.
  
  Minimum: {config, vs_payload, n_iterations,
            success, adversary_response, content_profile}

COMPONENT B: Principle Distillation  
  After a successful encoding, the agent generates
  an ABSTRACT GENERALIZABLE PRINCIPLE.
  Not "rli_split:4 worked on '48271039'" —
  that's too specific.
  But "for 8-digit numeric strings, splitting at position
  4 with RLI isolates defeats GPT models consistently" —
  that generalises to new strings.

COMPONENT C: Persistent Memory with Retrieval
  Principles are saved to disk (strategy_memory.json).
  When a new string arrives, relevant principles are
  retrieved based on content profile matching.
  The agent reads them before starting encoding.
  This is what makes Session 2 better than Session 1.
```

If any of these three is missing, the system is not
genuinely self-improving.

---

### What "Self-Improving" Looks Like in Practice

Here is the concrete behaviour you should observe
if the system is working correctly:

```
Document 1 (no memory):
  Agent prompt: [system prompt only]
  Iterations: 4
  Agent explores randomly, finds working config by trial

Document 10 (some memory):
  Agent prompt: [system prompt + 3 relevant principles]
  Iterations: 3
  Agent tries learned configs first, adapts faster

Document 50 (richer memory):
  Agent prompt: [system prompt + 5 relevant principles]
  Iterations: 2
  Agent has confident strategy for this content type

Document 100 (mature memory):
  Agent prompt: [system prompt + 5 confident principles]
  Iterations: 1
  Agent applies learned strategy directly, succeeds
```

The agent prompt gets RICHER over time as memory grows.
The agent's first attempt gets CLOSER to success over time.
This is the observable signature of self-improvement.

---

### The Self-Improving Loop in Full

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   New text arrives                                  │
│        │                                            │
│        ▼                                            │
│   Analyse structure (content profile)               │
│        │                                            │
│        ▼                                            │
│   Query strategy_memory.json                        │
│   Retrieve relevant past principles                 │
│        │                                            │
│        ▼                                            │
│   Inject principles into agent prompt               │
│        │                                            │
│        ▼                                            │
│   ┌──────────────────────────────────┐              │
│   │  Encoding loop                   │              │
│   │  Generate → Verify → Adversary   │              │
│   │  ↑                    │          │              │
│   │  └────────────────────┘          │              │
│   │  (iterate until success)         │              │
│   └──────────────────────────────────┘              │
│        │                                            │
│        ▼                                            │
│   On SUCCESS:                                       │
│   Distil abstract principle from this experience    │
│        │                                            │
│        ▼                                            │
│   Append to strategy_memory.json                    │
│   (persists to disk immediately)                    │
│        │                                            │
│        ▼                                            │
│   Ready for next text                               │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## PART 2: AUDIT THE EXISTING CODEBASE

Read every file in src/ and answer these questions
precisely. Do not guess. Look at the actual code.

---

### Question 1: Experience Capture

After each encoding attempt, is the following information
saved to disk (not just held in memory for the session)?

- [ ] The bidi config that was tried
- [ ] The VS payload that was tried
- [ ] Whether the adversary succeeded or failed
- [ ] The adversary's actual response text
- [ ] The content profile of the text being encoded
- [ ] The number of iterations taken
- [ ] The Hamming distance achieved
- [ ] The agent's own reasoning about why it failed

Report: YES (with file and line number) or NO for each.

---

### Question 2: Principle Distillation

After a successful encoding, does the system:

- [ ] Call an LLM to generate an abstract principle
- [ ] The principle mentions content TYPE not just config
- [ ] The principle would generalise to a new string
- [ ] The principle is saved to persistent storage

Report: YES (with file and line number) or NO for each.

---

### Question 3: Persistent Memory

- [ ] Does strategy_memory.json (or equivalent) exist?
- [ ] Is it written to disk after each success?
- [ ] Is it loaded at the START of each new document?
- [ ] Does it survive between Python process restarts?
- [ ] Is retrieval based on content profile matching?
      (not just returning all stored principles)

Report: YES (with file and line number) or NO for each.

---

### Question 4: Convergence Measurement

- [ ] Is iterations_to_convergence logged per document?
- [ ] Is document index tracked (document 1, 2, 3...)?
- [ ] Is there a mechanism to plot iterations vs doc index?

Report: YES (with file and line number) or NO for each.

---

### Question 5: Memory Injection Into Prompt

- [ ] Are retrieved principles injected into the agent
      prompt BEFORE the encoding loop starts?
- [ ] Does the prompt change based on what is in memory?
- [ ] Is there a LEARNED STRATEGIES section in the prompt?

Report: YES (with file and line number) or NO for each.

---

### Question 6: Level Classification

Based on your audit, classify the current system:

LEVEL 1: Output refinement only (no cross-document learning)
LEVEL 2: In-session accumulation (resets between sessions)
LEVEL 3: Genuine self-improvement (persists across sessions)

State which level it is and cite specific evidence.

---

## PART 3: WHAT TO IMPLEMENT IF MISSING

After the audit, implement ONLY what is missing.
Do not rewrite working components.

---

### If Experience Capture is missing:

Add to ghost_agent.py, inside the encoding loop,
after each adversary query:

```python
# Save attempt to disk immediately
experience = {
    "timestamp": datetime.now().isoformat(),
    "doc_index": doc_index,        # global counter
    "content_profile": analyse_structure(target),
    "bidi_config": current_bidi_config,
    "vs_payload": current_vs_payload,
    "hamming_dist": current_hamming,
    "adversary_model": adversary_model_id,
    "adversary_response": adversary_response,
    "extraction_succeeded": result["extracted"],
    "iteration": current_iteration,
    "agent_reasoning": agent_text[:500]
}
append_to_experience_log(experience)
# experience_log.jsonl — one JSON object per line
# JSONL not JSON so partial writes don't corrupt the file
```

---

### If Principle Distillation is missing:

Add to ghost_agent.py, after a successful encoding:

```python
DISTILLATION_PROMPT = """
You just successfully encoded text to defeat {adversary}.

Content profile:
  Type: {content_type}
  Length: {n_chars} characters
  Sentences: {n_sentences}
  Has numbers: {has_numbers}

What worked:
  Bidi config: {bidi_config}
  VS payload: {vs_payload}
  Iterations needed: {n_iterations}
  Hamming achieved: {hamming}/{total_len}

Write ONE abstract principle (max 40 words) that generalises
this success to similar content. It must mention:
  1. What content profile this applies to
  2. What strategy to try first
  3. Why it works mechanistically

Do not mention the specific string. Generalise.
"""

def distil_principle(success_data: dict) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": DISTILLATION_PROMPT.format(**success_data)
        }]
    )
    return response.content[0].text.strip()
```

---

### If Persistent Memory with Retrieval is missing:

Create src/strategy_memory.py:

```python
"""
src/strategy_memory.py

Persistent strategy memory for GHOST self-improving agent.

Storage:
  strategy_memory.json — indexed by content profile
  experience_log.jsonl — raw experience log (append-only)

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
import uuid
from datetime import datetime
from pathlib import Path
from ghost_tools_structural import tool_analyse_structure


MEMORY_PATH = Path("results/strategy_memory.json")
EXPERIENCE_LOG_PATH = Path("results/experience_log.jsonl")


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
    """Append one experience to the log. Never corrupts."""
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
        # Content type match is most important
        if p.get("content_type") == profile["content_type"]:
            score += 3.0
        # Length range match
        lo, hi = p.get("n_chars_range", [0, 9999])
        if lo <= profile["n_chars"] <= hi:
            score += 2.0
        elif abs(profile["n_chars"] - (lo+hi)/2) < (hi-lo):
            score += 1.0
        # Number presence match
        if p.get("has_numbers") == profile["has_numbers"]:
            score += 1.0
        # Confidence bonus
        confidence_bonus = {
            "high": 1.0, "medium": 0.5, "low": 0.0
        }
        score += confidence_bonus.get(
            p.get("confidence", "low"), 0.0
        )
        return score

    scored = sorted(
        principles,
        key=relevance_score,
        reverse=True
    )
    return [p["principle"] for p in scored[:top_k]
            if relevance_score(p) > 0]


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

    If a principle with matching profile already exists,
    increment its supporting_evidence counter and
    upgrade confidence if warranted.
    Otherwise create a new principle entry.
    """
    # Check if similar principle already exists
    n_chars = content_profile.get("n_chars", 0)

    for existing in memory["principles"]:
        if (existing["content_type"] ==
                content_profile.get("content_type") and
                abs(existing["n_chars_range"][0] - n_chars) < 50):
            # Strengthen existing
            existing["supporting_evidence"] += 1
            existing["last_updated"] = datetime.now().isoformat()
            # Upgrade confidence
            ev = existing["supporting_evidence"]
            existing["confidence"] = (
                "high" if ev >= 5 else
                "medium" if ev >= 2 else
                "low"
            )
            return memory

    # Add new principle
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
    Format retrieved principles for injection into agent prompt.
    Returns empty string if no principles.
    """
    if not principles:
        return ""

    lines = [
        "LEARNED STRATEGIES FROM PREVIOUS DOCUMENTS:",
        "Apply these before experimenting with new configs.",
        ""
    ]
    for i, p in enumerate(principles, 1):
        lines.append(f"{i}. {p}")

    lines.append("")
    return "\n".join(lines)
```

---

### If Convergence Measurement is missing:

Add to convergence.py:

```python
# Track per-document metrics with document index
convergence_log = []

for doc_idx, doc in enumerate(sampled, 1):
    for field_name, field_value in doc["fields"].items():
        result = run_ghost_agent(...)
        convergence_log.append({
            "doc_index": doc_idx,
            "domain": doc["domain"],
            "content_type": analyse_structure(
                field_value
            )["content_type"],
            "n_chars": len(field_value),
            "iterations": result["n_iterations"],
            "success": result["success"],
            "hamming": result["final_hamming"],
            "n_principles_available":
                len(retrieve_relevant_principles(
                    field_value, load_memory()
                ))
        })

# Save convergence log
with open("results/tables/convergence_log.csv", 'w') as f:
    writer = csv.DictWriter(f,
        fieldnames=convergence_log[0].keys())
    writer.writeheader()
    writer.writerows(convergence_log)

# This CSV produces Figure 2:
# X-axis: doc_index
# Y-axis: iterations
# Should trend downward as doc_index increases
```

---

### If Memory Injection is missing:

In ghost_agent.py, at the start of run_ghost_agent():

```python
def run_ghost_agent(target, field_name, ...):
    
    # Load memory and retrieve relevant principles
    memory = load_memory()
    principles = retrieve_relevant_principles(target, memory)
    principles_text = format_principles_for_prompt(principles)
    
    # Build system prompt WITH learned strategies injected
    system_prompt = GHOST_AGENT_SYSTEM_PROMPT
    if principles_text:
        system_prompt = (
            GHOST_AGENT_SYSTEM_PROMPT +
            "\n\n" + principles_text
        )
    
    # ... rest of agent loop using system_prompt
    # NOT the bare GHOST_AGENT_SYSTEM_PROMPT constant
```

---

## PART 4: WHAT TO REPORT BACK

After completing the audit and implementing missing
components, report the following:

1. CURRENT LEVEL: which level (1, 2, or 3) was the
   system at before your changes?

2. EVIDENCE: for each of the 5 questions, what was
   present and what was absent?

3. CHANGES MADE: list every file modified and every
   function added, with a one-line description.

4. SMOKE TEST RESULTS: run this sequence and report
   the output:

   # Process 3 documents with memory enabled
   python src/convergence.py \
     --n_samples 3 \
     --max_iterations 3

   # Check memory was written
   cat results/strategy_memory.json | python -m json.tool

   # Process 3 more documents
   python src/convergence.py \
     --n_samples 3 \
     --max_iterations 3 \
     --start_index 3

   # Verify memory grew
   # Verify second run used principles from first run
   # Report: did iterations decrease in second run?

5. HONEST ASSESSMENT: after your changes, is the system
   genuinely at Level 3? What evidence supports this?
   What is still missing?

Do not claim Level 3 without the convergence curve
showing decreasing iterations over document index.
That curve is the proof.
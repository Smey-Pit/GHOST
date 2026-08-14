"""
src/bidi_permute.py

Prototype: automatic bidi-permutation encoding for GHOST Mechanism 2.

Given a field value (e.g. an account number), finds a SEPARABLE permutation
of its digits (the class of permutations realizable by nested Unicode
bidi isolates/overrides -- permutations avoiding patterns 2413 and 3142)
that is maximally different (Hamming distance) from the original, then
constructs the RLI/RLO/LRI/PDI/PDF bracket string that renders back to the
original when displayed, while storing/tokenizing as the far permutation.

Requires `python-bidi` (wraps the Rust unicode-bidi crate) for the
oracle used in __main__ verification -- this is the same UAX#9
implementation real renderers use, not a hand-rolled approximation.
"""
import random
import string
from bidi import get_display

RLI, LRI, FSI, PDI = '⁧', '⁦', '⁨', '⁩'
RLO, LRO, PDF = '‮', '‭', '‬'
CTRL = {RLI, LRI, FSI, PDI, RLO, LRO, PDF}


def visible(s):
    return ''.join(c for c in get_display(s) if c not in CTRL)


# ---------- 1. separable decomposition ----------

def _try_kind(perm, kind):
    n = len(perm)
    blocks, i = [], 0
    ptr = min(perm) if kind == 'sum' else max(perm)
    while i < n:
        grown = False
        for k in range(1, n - i + 1):
            chunk = perm[i:i + k]
            lo, hi = min(chunk), max(chunk)
            if hi - lo + 1 != k or set(chunk) != set(range(lo, hi + 1)):
                continue
            if kind == 'sum' and lo != ptr:
                continue
            if kind == 'skew' and hi != ptr:
                continue
            blocks.append(chunk)
            i += k
            ptr = (hi + 1) if kind == 'sum' else (lo - 1)
            grown = True
            break
        if not grown:
            return None
    return blocks if len(blocks) >= 2 else None


def decompose(perm):
    n = len(perm)
    if n == 1:
        return ('leaf', perm[0])
    for kind in ('sum', 'skew'):
        blocks = _try_kind(perm, kind)
        if blocks:
            children = [decompose(b) for b in blocks]
            if any(c is None for c in children):
                return None
            return (kind, children)
    return None


def is_separable(perm):
    return decompose(tuple(perm)) is not None


# ---------- 2. tree -> bidi string ----------

def _leaves_are_pure_run(tree):
    kind = tree[0]
    if kind == 'leaf':
        return True
    return all(c[0] == 'leaf' for c in tree[1])


def encode(tree, chars):
    kind = tree[0]
    if kind == 'leaf':
        return chars[tree[1]]
    children = tree[1]
    if kind == 'sum':
        # no reordering needed among siblings; bare leaves are safe to concatenate
        return ''.join(
            encode(c, chars) if c[0] == 'leaf' else _wrap_child(c, chars)
            for c in children
        )
    # kind == 'skew': siblings must be individually reordered by the parent
    # RTL context, which only works on ISOLATED atomic runs -- adjacent bare
    # leaves would merge into one un-reorderable digit run, so every child
    # (leaves included) must get its own isolate here.
    if _leaves_are_pure_run(tree):
        run = ''.join(chars[c[1]] for c in children)
        return RLO + run + PDF
    parts = [
        _wrap_child(c, chars) if c[0] != 'leaf' else (LRI + chars[c[1]] + PDI)
        for c in children
    ]
    return RLI + ''.join(parts) + PDI


def _wrap_child(tree, chars):
    inner = encode(tree, chars)
    wrapper = LRI if tree[0] == 'sum' else RLI
    return wrapper + inner + PDI


def encode_permutation(original, stored_order):
    tree = decompose(tuple(stored_order))
    if tree is None:
        return None
    return encode(tree, original)


# ---------- 3. search for a target permutation ----------

def random_separable_perm(n, rng):
    if n == 1:
        return [0]
    k = rng.randint(1, n - 1)
    left = random_separable_perm(k, rng)
    right = random_separable_perm(n - k, rng)
    if rng.random() < 0.5:
        return left + [x + k for x in right]
    else:
        return [x + (n - k) for x in left] + right


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def find_target_permutation(field, trials=2000, seed=0):
    """
    Search for the separable permutation of `field` maximizing Hamming
    distance from identity, SUBJECT TO round-trip correctness: a
    candidate is only considered if encode_permutation(field, perm)
    actually renders back to `field` via the real UAX#9 oracle.

    This filter is required for general text -- it was never needed for
    the digit-only fields this module was originally built for (their
    round-trip always held), but real prose contains neutral/weak
    bidi-type characters (spaces, punctuation) whose UAX#9 resolution
    (W1-W7/N1-N2 neutral rules) can differ from the naive "every
    character is an independently reorderable opaque unit" assumption
    this module's construction makes, DEPENDING on where that neutral
    character lands in the specific permutation chosen -- confirmed for
    real (src/bidi_permute.py's sentence-scale self-test measured a real
    ~10-30% failure rate on realistic sentence-like text before this
    filter existed, purely because the search only ever optimized
    Hamming distance and never verified the thing it was supposed to
    guarantee). Field-length digit content is unaffected in practice
    (round-trip already always held there) but now gets verified
    instead of assumed, at the cost of a real bidi.get_display() oracle
    call per trial instead of a cheap integer comparison.

    Returns (None, None) if no candidate round-trips within `trials` --
    callers must handle this (see encode_field).
    """
    n = len(field)
    rng = random.Random(seed)
    identity = list(range(n))
    best = None
    for _ in range(trials):
        perm = random_separable_perm(n, rng)
        enc = encode_permutation(field, perm)
        if enc is None or visible(enc) != field:
            continue
        d = hamming(perm, identity)
        if best is None or d > best[0]:
            best = (d, perm)
    if best is None:
        return None, None
    return best


def encode_field(field, trials=2000, seed=0):
    """
    End-to-end: find the farthest separable permutation of `field`'s digits
    and return (encoded_string, stored_order_string, hamming_distance).

    Returns (None, None, 0) if the search exhausted `trials` without
    finding a permutation that round-trips (see find_target_permutation) --
    callers must treat this the same as any other invalid-encoding case
    (mirroring encode_permutation's existing None-on-failure convention).
    """
    dist, perm = find_target_permutation(field, trials=trials, seed=seed)
    if perm is None:
        return None, None, 0
    enc = encode_permutation(field, perm)
    stored = ''.join(field[p] for p in perm)
    return enc, stored, dist


def compare(field, trials=2000, seed=0):
    """
    Side-by-side of the two readings that matter for the threat model:
      - human / bidi-aware renderer (browser, PDF viewer, terminal): sees
        the correctly reordered digits, identical to the original.
      - naive extractor (scraper/LLM that reads codepoints in logical/
        stored order and just drops invisible control characters, without
        running the bidi algorithm): sees `stored`, a wrong-but-plausible
        permutation of the same character multiset.
    Returns a dict; see format_comparison() for a printable version.
    """
    enc, stored, dist = encode_field(field, trials=trials, seed=seed)
    rendered = visible(enc)
    diff_mask = ''.join(
        ' ' if a == b else '^' for a, b in zip(field, stored)
    )
    return {
        'original': field,
        'human_sees': rendered,
        'naive_extractor_sees': stored,
        'hamming_distance': dist,
        'length': len(field),
        'diff_mask': diff_mask,
        'same_multiset': sorted(field) == sorted(stored),
        'round_trip_ok': rendered == field,
        'encoded': enc,
    }


def format_comparison(field, trials=2000, seed=0):
    r = compare(field, trials=trials, seed=seed)
    lines = [
        f"original              : {r['original']}",
        f"human / renderer sees : {r['human_sees']}   (matches original: {r['round_trip_ok']})",
        f"naive extractor sees  : {r['naive_extractor_sees']}   (wrong digits: {r['diff_mask']})",
        f"hamming distance      : {r['hamming_distance']}/{r['length']}",
        f"same char multiset    : {r['same_multiset']}  "
        f"(naive read can't be flagged by a digit-histogram check)",
    ]
    return '\n'.join(lines)


def _self_test(quiet=False):
    """Regression check against the real UAX#9 oracle for field lengths 2-12."""
    rng = random.Random(42)
    ok, total = 0, 0
    for n in range(2, 13):
        for _ in range(30):
            field = ''.join(rng.choice('0123456789') for _ in range(n))
            enc, stored, dist = encode_field(field, trials=500, seed=rng.randint(0, 10**6))
            total += 1
            if enc is None or visible(enc) != field:
                print(f"FAIL n={n} field={field} -> visible={visible(enc) if enc else None!r}")
                continue
            ok += 1
    print(f"{ok}/{total} verified correct against real UAX#9 oracle")
    return ok == total


# ---------- 4. sentence-scale de-risking (Phase 1, sentence-level obfuscation) ----------

# UAX#9 caps explicit embedding levels at 125 -- a real hard limit real
# renderers implement, not a soft guideline. encode()'s recursive isolate/
# override nesting for an unbalanced decompose() tree could plausibly
# approach or exceed this at sentence length (never exercised before --
# every prior use of this module was on 2-12 char fields). This has never
# been measured anywhere in this codebase before this self-test.
MAX_UAX9_EMBEDDING_DEPTH = 125

_PUSH = {RLI, LRI, FSI, RLO, LRO}
_POP = {PDI, PDF}


def max_nesting_depth(encoded: str) -> int:
    """
    Approximate the peak bidi embedding depth of an encoded string by
    tracking a running counter: +1 on any isolate/override-opening control
    (RLI/LRI/FSI/RLO/LRO), -1 on any closing control (PDI/PDF). encode()
    constructs these in properly matched (isolate-opener -> PDI,
    override-opener -> PDF) pairs by design, so a single counter is a
    faithful proxy for the real embedding level UAX#9 tracks, without
    needing to reimplement the full bidi algorithm's paragraph-level
    resolution just to measure nesting depth.
    """
    depth = 0
    peak = 0
    for c in encoded:
        if c in _PUSH:
            depth += 1
            peak = max(peak, depth)
        elif c in _POP:
            depth -= 1
    return peak


def _random_sentence_like(rng, n):
    """
    Realistic sentence-shaped text for the scale test -- letters, spaces,
    and punctuation, not just a digit alphabet like _self_test() uses.
    Approximates real carrier sentences (e.g. "The value is 5926847003."
    from dataset.py's phrase templates, or Track A's narrative prose).
    """
    alphabet = string.ascii_letters + string.digits + ' .,-'
    # Weight space more heavily so word-like chunks emerge, closer to real
    # prose than a uniform character draw.
    weighted = alphabet + ' ' * 8
    return ''.join(rng.choice(weighted) for _ in range(n))


def _self_test_sentence_scale(lengths=(20, 50, 100, 200, 400), trials=500, quiet=False):
    """
    De-risks bidi_permute at sentence length BEFORE any pipeline wiring
    (see plan: sentence-level obfuscation, Phase 1 gate). Three checks per
    trial, all against the real UAX#9 oracle/limit, none against a mock:
      1. search quality -- Hamming distance achieved vs. length, at the
         SAME trial budget (500) already used in production (config.yaml's
         bidi_permute_search_trials)
      2. round-trip fidelity -- visible(encode(...)) == original, via the
         real python-bidi oracle (same check _self_test() already does)
      3. peak embedding depth vs. MAX_UAX9_EMBEDDING_DEPTH (125) -- the
         one dimension _self_test() never measured because it never
         mattered at field length
    """
    rng = random.Random(1337)
    results = []
    worst_depth = 0
    all_round_trip_ok = True
    for n in lengths:
        for trial_i in range(10):
            text = _random_sentence_like(rng, n)
            enc, stored, dist = encode_field(text, trials=trials, seed=rng.randint(0, 10**6))
            round_trip_ok = enc is not None and visible(enc) == text
            depth = max_nesting_depth(enc) if enc is not None else -1
            worst_depth = max(worst_depth, depth)
            all_round_trip_ok = all_round_trip_ok and round_trip_ok
            results.append({
                'n': n, 'trial': trial_i, 'hamming': dist,
                'hamming_ratio': dist / n if n else 0.0,
                'round_trip_ok': round_trip_ok, 'depth': depth,
            })
            if not round_trip_ok and not quiet:
                print(f"FAIL round-trip n={n} trial={trial_i} text={text!r}")

    if not quiet:
        for n in lengths:
            rows = [r for r in results if r['n'] == n]
            mean_ratio = sum(r['hamming_ratio'] for r in rows) / len(rows)
            max_depth_n = max(r['depth'] for r in rows)
            print(f"n={n:4d}: mean hamming_ratio={mean_ratio:.3f}  "
                  f"max_depth={max_depth_n}  round_trip_ok={all(r['round_trip_ok'] for r in rows)}")
        print(f"\nWorst embedding depth observed: {worst_depth} "
              f"(UAX#9 limit: {MAX_UAX9_EMBEDDING_DEPTH})")
        print(f"All round-trips OK: {all_round_trip_ok}")

    depth_ok = worst_depth < MAX_UAX9_EMBEDDING_DEPTH
    passed = all_round_trip_ok and depth_ok
    if not quiet:
        print(f"\nDECISION GATE: {'PASS' if passed else 'FAIL'} "
              f"(round_trip_ok={all_round_trip_ok}, depth_ok={depth_ok})")
    return {'passed': passed, 'results': results, 'worst_depth': worst_depth}


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == '--selftest':
        _self_test()
    elif args and args[0] == '--selftest-sentence':
        result = _self_test_sentence_scale()
        sys.exit(0 if result['passed'] else 1)
    elif args and args[0] == '--compare':
        print(format_comparison(args[1]))
    elif args:
        field = args[0]
        enc, stored, dist = encode_field(field)
        print(f"original       : {field}")
        print(f"stored/logical : {stored}  (hamming distance {dist}/{len(field)})")
        print(f"codepoints     : {' '.join(hex(ord(c)) for c in enc)}")
        print(f"renders as     : {visible(enc)}")
        print(f"round-trip ok  : {visible(enc) == field}")
    else:
        _self_test()

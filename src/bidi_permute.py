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
    n = len(field)
    rng = random.Random(seed)
    identity = list(range(n))
    best = None
    for _ in range(trials):
        perm = random_separable_perm(n, rng)
        d = hamming(perm, identity)
        if best is None or d > best[0]:
            best = (d, perm)
    return best


def encode_field(field, trials=2000, seed=0):
    """
    End-to-end: find the farthest separable permutation of `field`'s digits
    and return (encoded_string, stored_order_string, hamming_distance).
    """
    dist, perm = find_target_permutation(field, trials=trials, seed=seed)
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


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    if args and args[0] == '--selftest':
        _self_test()
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

"""Schedule = an ordered list of loop transformations.

Text DSL (one transform per line, applied top to bottom):

    reorder(i, k, j)     # permute the nest into this loop order
    tile(i, 32)          # split loop i into i_t (step 32) + i (point loop)
    parallel(i)          # #pragma omp parallel for on loop i
    unroll(k, 4)         # #pragma clang loop unroll_count(4) on loop k

Var names refer to the kernel's loop variables (and to tile loops once created,
e.g. `i_t`). This mirrors Tiramisu's schedule API closely enough for an LLM to
emit, while staying trivial to parse.
"""
import re

_LINE = re.compile(r"\s*(\w+)\s*\(([^)]*)\)\s*")

# The 9-primitive repertoire (Tiramisu/ComPilot), plus reorder as generalized interchange.
_PRIMITIVES = {
    "interchange", "reorder", "parallel", "tile", "tile2d", "tile3d",
    "unroll", "skew", "reverse", "fuse", "shift",
}


def parse(text: str):
    """Parse schedule text into [(op, [args...]), ...]. Raises ValueError."""
    ops = []
    for raw in text.strip().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE.fullmatch(line)
        if not m:
            raise ValueError(f"cannot parse schedule line: {raw!r}")
        op = m.group(1).lower()
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if op not in _PRIMITIVES:
            raise ValueError(f"unknown transform {op!r} (known: {sorted(_PRIMITIVES)})")
        ops.append((op, args))
    return ops

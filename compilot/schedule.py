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
# Every arg is either a loop identifier or an integer factor. Enforcing this stops
# anything else (these args are interpolated into the C that gets compiled and run).
_ARG = re.compile(r"-?\d+|[A-Za-z_]\w*")

# The 9-primitive repertoire (Tiramisu/ComPilot), plus reorder as generalized interchange.
_PRIMITIVES = {
    "interchange", "reorder", "parallel", "tile", "tile2d", "tile3d",
    "unroll", "skew", "reverse", "fuse", "shift",
}

# Arg positions that are loop SIZE factors (tile blocks / unroll counts). These become
# C loop steps, so they must be positive integers -- 0 or negative makes the generated
# loop never advance and spin until the run timeout. (skew coefficients may be negative,
# so they are deliberately not listed here.)
_FACTOR_POS = {"tile": (1,), "unroll": (1,), "tile2d": (2, 3), "tile3d": (3, 4, 5)}


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
        for a in args:
            if not _ARG.fullmatch(a):
                raise ValueError(f"invalid schedule argument {a!r} in {raw!r}")
        for pos in _FACTOR_POS.get(op, ()):
            val = args[pos] if pos < len(args) else ""
            if not val.isdigit() or int(val) == 0:   # .isdigit() rejects '-16' and loop names
                raise ValueError(f"{op} factor must be a positive integer, got {args!r} in {raw!r}")
        ops.append((op, args))
    return ops

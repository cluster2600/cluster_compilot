"""End-to-end environment test: legality + real measured speedup on GEMM.

Proves the agent's environment works without any LLM: legal schedules compile,
run, and report a real speedup; illegal ones are rejected before execution.
Run: python3 -m tests.test_environment
"""
from compilot.kernels import GEMM
from compilot.backend_isl import Environment
from tests.test_legality import GEMM as GEMM_POLY

env = Environment(GEMM, GEMM_POLY)

CASES = [
    ("baseline (identity)",        ""),
    ("reorder(i,k,j)",             "reorder(i, k, j)"),
    ("tile2d + parallel",          "reorder(i, k, j)\ntile2d(i, j, 32, 32)\nparallel(i_t)"),
    ("reverse(k)  [illegal]",      "reverse(k)"),
    ("parallel(k) [illegal]",      "parallel(k)"),
]

if __name__ == "__main__":
    print(f"baseline time: {env.baseline()['time']:.4f}s\n")
    for name, sched in CASES:
        r = env.evaluate(sched)
        sp = f"{r.speedup:.2f}x" if r.speedup else "  -  "
        print(f"[{r.status:16}] {sp:>7}  {name}")
        if r.status not in ("success", "illegal", "parallel_illegal"):
            print(f"                  detail: {r.detail[:120]}")

"""End-to-end environment test: legality + real measured speedup on GEMM.

Proves the agent's environment works without any LLM: legal schedules compile,
run, and report a real speedup; illegal ones are rejected before execution.
Run: python3 -m tests.test_environment
"""
from compilot.backend_isl import environment

env = environment("gemm")

# (name, schedule, expected status). Legal cases must run AND report a real speedup;
# illegal ones must be rejected by the legality engine BEFORE execution.
CASES = [
    ("baseline (identity)",        "",                                                    "success"),
    ("reorder(i,k,j)",             "reorder(i, k, j)",                                    "success"),
    ("tile2d + parallel",          "reorder(i, k, j)\ntile2d(i, j, 32, 32)\nparallel(i_t)", "success"),
    ("reverse(k)  [illegal]",      "reverse(k)",                                          "illegal"),
    ("parallel(k) [illegal]",      "parallel(k)",                                         "parallel_illegal"),
]


def test_environment():
    for name, sched, want in CASES:
        r = env.evaluate(sched)
        assert r.status == want, f"{name}: got {r.status} ({r.detail[:120]}), want {want}"
        if want == "success":
            assert r.speedup and r.speedup > 0, f"{name}: legal schedule had no speedup"
        print(f"OK [{r.status:16}] {(f'{r.speedup:.2f}x' if r.speedup else '  -  '):>7}  {name}")


if __name__ == "__main__":
    print(f"baseline time: {env.baseline()['time']:.4f}s\n")
    test_environment()
    print("\ntest_environment: all assertions passed")

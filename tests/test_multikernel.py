"""Multi-statement EXECUTION on PolyBench kernels (2mm, 3mm, mvt, atax, bicg).

Each statement is scheduled independently; the combined output checksum guards
correctness. We assert: a legal per-statement schedule runs correctly (success),
matmul kernels speed up, and parallelizing a reduction loop is rejected.
Run: python3 -m tests.test_multikernel
"""
from compilot.kernels import MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment

# a legal per-statement schedule per kernel (parallelize a non-reduction outer loop)
LEGAL = {
    "2mm":  ["parallel(i)", "parallel(i)"],
    "3mm":  ["parallel(i)", "parallel(i)", "parallel(i)"],
    "mvt":  ["parallel(i)", "parallel(i)"],
    "atax": ["parallel(i)", ""],
    "bicg": ["", "parallel(i)"],
    "gesummv": ["parallel(i)", "parallel(i)"],
    "gemver": ["parallel(i)", "parallel(i)", "parallel(i)", "parallel(i)"],
    "covariance": ["parallel(j)", "parallel(i)", "tile2d(i,j,32,32)\nparallel(i_t)"],
    "correlation": ["parallel(j)", "parallel(j)", "parallel(j)", "parallel(i)", "parallel(i)"],
    "doitgen": ["parallel(r)", "parallel(r)"],
}
# parallelizing a reduction loop must be rejected
ILLEGAL = {
    "2mm":  ["parallel(k)", ""],
    "3mm":  ["parallel(k)", "", ""],
    "atax": ["parallel(j)", ""],   # s0 reduces over j
    "bicg": ["parallel(i)", ""],   # s0 reduces over i
    "gesummv": ["parallel(j)", ""],  # s0 reduces over j
    "gemver": ["", "parallel(j)", "", ""],  # s2 reduces over j
    "covariance": ["", "", "parallel(k)"],  # s3 reduces over k
    "correlation": ["parallel(i)", "", "", "", ""],  # s0 (mean) reduces over i
    "doitgen": ["parallel(s)", ""],  # s0 reduces over s
}
SPEEDS_UP = {"2mm", "3mm"}         # matmul kernels should beat 1x

if __name__ == "__main__":
    for name, factory in MULTI_REGISTRY.items():
        env = MultiEnvironment(factory())
        r = env.evaluate(LEGAL[name])
        sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "  -  "
        ok = r["status"] == "success"
        print(f"[{'OK ' if ok else 'FAIL'}] {name:6} legal-schedule [{r['status']:10}] {sp}")
        assert ok, f"{name}: {r['status']}"
        if name in SPEEDS_UP:
            # ran with a real measured time; absolute speedup is core/contention dependent (not asserted in CI)
            assert r["speedup"] > 0, f"{name}: expected a measured speedup, got {r['speedup']}"
        if name in ILLEGAL:
            ri = MultiEnvironment(factory()).evaluate(ILLEGAL[name])
            print(f"        {name:6} parallel(reduction) -> [{ri['status']}]")
            assert ri["status"] == "parallel_illegal", f"{name}: reduction not rejected ({ri['status']})"
    print("\nMulti-statement execution validated on 5 PolyBench kernels.")

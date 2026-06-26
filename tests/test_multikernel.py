"""Multi-statement EXECUTION on 2mm (tmp = A·B; D = tmp·C).

Each matmul statement is scheduled independently; the final checksum guards
correctness. Proves: independent scheduling speeds up the whole kernel, and
the engine rejects parallelizing a reduction loop.
Run: python3 -m tests.test_multikernel
"""
from compilot.kernels import MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment

env = MultiEnvironment(MULTI_REGISTRY["2mm"]())

CASES = [
    ("identity", ["", ""], "success", False),   # ~1.0x (same code as baseline; timing noise)
    ("both tiled+parallel",
     ["reorder(i,k,j)\ntile2d(i,j,64,64)\nparallel(i_t)",
      "reorder(i,l,j)\ntile2d(i,j,64,64)\nparallel(i_t)"], "success", True),
    ("S1 parallel(l) [reduction]", ["", "parallel(l)"], "parallel_illegal", False),
]

if __name__ == "__main__":
    print(f"2mm baseline: {env.baseline()['time']:.4f}s")
    for name, scheds, want_status, want_speedup in CASES:
        r = env.evaluate(scheds)
        ok = r["status"] == want_status
        sp = f"{r['speedup']:.2f}x" if r["speedup"] else "  -  "
        print(f"[{'OK ' if ok else 'FAIL'}] {r['status']:16} {sp:>7}  {name}")
        assert ok, f"{name}: {r['status']} != {want_status}"
        if want_speedup:
            assert r["speedup"] and r["speedup"] > 1.0, f"{name}: expected speedup"
    print("\nMulti-statement execution (2mm) validated.")

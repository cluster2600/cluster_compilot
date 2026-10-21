"""PolyBench coverage extras: the ×5 dataset classes and the correlation kernel.

    python3 -m tests.test_polybench
"""
from compilot.kernels import (REGISTRY, MULTI_REGISTRY, STENCIL_REGISTRY, IMPERFECT_REGISTRY,
                              POLYBENCH, POLYBENCH_DATASETS, SIZE_CLASSES, SIZE_NAMES, sized_kernel)
from compilot.mcp_server import _optimize


def _sizes(name, size):
    k = sized_kernel(name, size)
    return k[0].sizes if isinstance(k, tuple) else k.sizes


def test_size_classes_are_polybench_datasets():
    # DEFAULT is the registry's own size; MINI..EXTRALARGE are PolyBench 4.2.1's tables
    assert _sizes("gemm", "DEFAULT") == REGISTRY["gemm"][0].sizes
    assert _sizes("gemm", "LARGE") == {"N": 1000, "M": 1100, "K": 1200}      # NI, NJ, NK
    assert _sizes("fdtd2d", "MINI") == {"NX": 20, "NY": 30, "TSTEPS": 20}     # TMAX -> TSTEPS
    names = [*REGISTRY, *MULTI_REGISTRY, *STENCIL_REGISTRY, *IMPERFECT_REGISTRY]
    n = 0
    for name in names:
        for size in SIZE_NAMES:
            got = _sizes(name, size)                 # every kernel builds at every class
            if name in POLYBENCH and size != "DEFAULT":
                pb, rename = POLYBENCH[name]
                for ours, theirs in rename.items():
                    assert got[ours] == POLYBENCH_DATASETS[pb][size][theirs], (name, size, ours)
                n += 1
    assert set(POLYBENCH) | {"distmatrix", "montecarlo"} == set(names), set(names) - set(POLYBENCH)
    # the two non-PolyBench extras scale their DEFAULT sizes instead
    base = _sizes("distmatrix", "DEFAULT")
    assert _sizes("distmatrix", "MINI") == {k: max(2, round(v * SIZE_CLASSES["MINI"])) for k, v in base.items()}
    print(f"OK: {n} kernel x class instances match PolyBench/C 4.2.1's dataset tables")


def test_correlation_registered_and_runs():
    # Deterministic end-to-end: legality engine + codegen handle all 5 statements
    # (mean, variance, sqrt+guard stddev, in-place normalize, triangular matmul).
    # parallel(i) on the corr matmul (stmt 4) is independent across rows -> legal.
    from compilot.multikernel import MultiEnvironment
    assert "correlation" in MULTI_REGISTRY
    menv = MultiEnvironment(sized_kernel("correlation", "SMALL"))
    r = menv.evaluate(["", "", "", "", "parallel(i)"])
    assert r["status"] == "success", r          # legal + checksum matches baseline
    assert r["speedup"] is not None and r["speedup"] > 0, r
    print(f"OK: correlation SMALL parallel(corr) -> {r['status']}, {r['speedup']:.2f}x")


def test_optimize_respects_size_class():
    # a MINI gemm run should still produce a legal, measured schedule
    r = _optimize("gemm", backend="mock", iters=4, size="MINI")
    assert r["best_speedup"] >= 1.0, r
    print(f"OK: optimize(gemm, mock, MINI) -> {r['best_speedup']:.2f}x")


if __name__ == "__main__":
    test_size_classes_are_polybench_datasets()
    test_correlation_registered_and_runs()
    test_optimize_respects_size_class()
    print("test_polybench: all checks passed")

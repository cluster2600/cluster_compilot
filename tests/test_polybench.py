"""PolyBench coverage extras: the ×5 size classes and the correlation kernel.

    python3 -m tests.test_polybench
"""
from compilot.kernels import (REGISTRY, MULTI_REGISTRY, STENCIL_REGISTRY,
                              SIZE_CLASSES, sized_kernel)
from compilot.mcp_server import _optimize


def test_size_classes_scale_bounds():
    # every class scales every loop bound monotonically; LARGE == the registry default.
    base = dict(REGISTRY["gemm"][0].sizes)
    assert sized_kernel("gemm", "LARGE")[0].sizes == base
    mini = sized_kernel("gemm", "MINI")[0].sizes
    xl = sized_kernel("gemm", "EXTRALARGE")[0].sizes
    for k in base:
        assert mini[k] < base[k] < xl[k], (k, mini[k], base[k], xl[k])
    # works for multi + stencil kernels too (mutates a fresh object, no shared state)
    assert sized_kernel("covariance", "SMALL").sizes["N"] < sized_kernel("covariance", "LARGE").sizes["N"]
    assert sized_kernel("jacobi2d", "MINI").sizes["TSTEPS"] >= 2
    print(f"OK: size classes {sorted(SIZE_CLASSES)} scale gemm "
          f"{mini['N']}<{base['N']}<{xl['N']}")


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
    test_size_classes_scale_bounds()
    test_correlation_registered_and_runs()
    test_optimize_respects_size_class()
    print("test_polybench: all checks passed")

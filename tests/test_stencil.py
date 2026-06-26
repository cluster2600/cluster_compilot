"""Stencils: a sequential time loop wrapping scheduled spatial sweeps.

jacobi-1d / jacobi-2d: each time step writes one buffer from the other (no spatial
self-dependence), so the spatial loops parallelize; the time loop stays sequential.
Validates correct execution + legal spatial scheduling.
Run: python3 -m tests.test_stencil
"""
from compilot.kernels import STENCIL_REGISTRY
from compilot.stencil import StencilEnvironment

CASES = {
    "jacobi1d": ["parallel(i)", "parallel(i)"],
    "jacobi2d": ["tile2d(i,j,64,64)\nparallel(i_t)", "tile2d(i,j,64,64)\nparallel(i_t)"],
}

if __name__ == "__main__":
    for name, factory in STENCIL_REGISTRY.items():
        env = StencilEnvironment(factory())
        r = env.evaluate(CASES[name])
        sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "  -  "
        ok = r["status"] == "success"
        print(f"[{'OK ' if ok else 'FAIL'}] {name:10} baseline={env.baseline()['time']:.4f}s "
              f"[{r['status']:10}] {sp}")
        assert ok, f"{name}: {r['status']}"
    print("\nStencil time-loop kernels validated (jacobi-1d, jacobi-2d).")

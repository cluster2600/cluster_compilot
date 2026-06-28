"""Stencils: a sequential time loop wrapping scheduled spatial sweeps.

jacobi-1d / jacobi-2d: each time step writes one buffer from the other (no spatial
self-dependence), so the spatial loops parallelize; the time loop stays sequential.
Validates correct execution + legal spatial scheduling.
Run: python3 -m tests.test_stencil
"""
from compilot.kernels import STENCIL_REGISTRY
from compilot.stencil import StencilEnvironment
from compilot import prompt

CASES = {
    "jacobi1d": ["parallel(i)", "parallel(i)"],
    "jacobi2d": ["tile2d(i,j,64,64)\nparallel(i_t)", "tile2d(i,j,64,64)\nparallel(i_t)"],
    "seidel2d": ["skew(j,i,1)"],   # jacobi parallelizes; seidel needs skewing
    "heat3d": ["parallel(i)", "parallel(i)"],   # 3-D Jacobi: spatial loops fully parallel
    # fdtd-2d: source boundary + 3 field updates, every sweep reads other arrays -> parallel
    "fdtd2d": ["parallel(j)", "parallel(i)", "parallel(i)", "parallel(i)"],
    # adi: column then row sweep (each forward + back); parallel on the orthogonal i
    "adi": ["parallel(i)"] * 8,
    # deriche: horizontal passes parallel across rows (i); vertical across columns (j)
    "deriche": ["parallel(i)", "parallel(i)", "parallel(i)",
                "parallel(j)", "parallel(j)", "parallel(j)"],
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
    # seidel carries spatial dependences: naive parallelism must be rejected
    se = StencilEnvironment(STENCIL_REGISTRY["seidel2d"]())
    for bad in (["parallel(i)"], ["parallel(j)"]):
        st = se.evaluate(bad)["status"]
        print(f"        seidel2d {bad[0]} -> [{st}]")
        assert st == "parallel_illegal", f"seidel {bad}: {st}"
    # regression: the dialogue prompt builder must handle SStmt (no `.output`) — used to
    # crash run_agent.py --kernel jacobi2d with AttributeError in kernel_message_multi.
    for name, factory in STENCIL_REGISTRY.items():
        msg = prompt.kernel_message_multi(StencilEnvironment(factory()))
        assert "SEQUENTIAL" in msg and "<schedule>" in msg, f"{name}: stencil prompt malformed"
    print("        prompt builder OK for all stencils (no SStmt.output crash)")
    print("\nStencils validated: jacobi-1d/2d parallel; seidel-2d needs skewing (parallel rejected).")

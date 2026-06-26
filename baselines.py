"""Baseline comparison: ComPilot vs naive auto-parallelization.

The paper compares ComPilot against the Pluto polyhedral optimizer. Pluto's
bundled piplib does not compile on this toolchain (Darwin-27 / clang-22 rejects
its legacy K&R C; the LLVM-14 clang++ can't link C++), so we report a tractable
*proxy* baseline instead: NAIVE = parallelize the outermost non-reduction loop
only (a simple compiler heuristic), with no tiling/reorder/unroll. The ratio
shows how much ComPilot's polyhedral-guided schedule beats naive parallelism.

Run: python3 baselines.py
"""
import math

from compilot.backend_isl import environment
from compilot.kernels import MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment
from bench import SINGLE, MULTI            # ComPilot's strong schedules

# NAIVE: parallelize the outermost non-reduction loop only
NAIVE_SINGLE = {"gemm": "parallel(i)", "syrk": "parallel(i)",
                "syr2k": "parallel(i)", "floydwarshall": ""}
NAIVE_MULTI = {"2mm": ["parallel(i)", "parallel(i)"], "3mm": ["parallel(i)"] * 3,
               "mvt": ["parallel(i)", "parallel(i)"], "atax": ["parallel(i)", ""],
               "bicg": ["", "parallel(i)"], "gesummv": ["parallel(i)", "parallel(i)"]}


def sp_single(name, sched):
    r = environment(name).evaluate(sched)
    return r.speedup if r.status == "success" else 1.0


def sp_multi(name, scheds):
    r = MultiEnvironment(MULTI_REGISTRY[name]()).evaluate(scheds)
    return r["speedup"] if r["status"] == "success" else 1.0


def main():
    print(f"{'kernel':16}{'naive':>9}{'ComPilot':>11}{'ratio':>9}")
    print("-" * 46)
    naive_all, comp_all = [], []
    for name in SINGLE:
        n, c = sp_single(name, NAIVE_SINGLE[name]), sp_single(name, SINGLE[name])
        naive_all.append(n); comp_all.append(c)
        print(f"{name:16}{n:8.2f}x{c:10.2f}x{c/n:8.2f}x")
    for name in MULTI:
        n, c = sp_multi(name, NAIVE_MULTI[name]), sp_multi(name, MULTI[name])
        naive_all.append(n); comp_all.append(c)
        print(f"{name:16}{n:8.2f}x{c:10.2f}x{c/n:8.2f}x")
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    print("-" * 46)
    print(f"{'GEOMEAN':16}{gm(naive_all):8.2f}x{gm(comp_all):10.2f}x{gm(comp_all)/gm(naive_all):8.2f}x")
    print(f"\nComPilot is {gm(comp_all)/gm(naive_all):.2f}x faster than naive auto-parallelization (geomean).")


if __name__ == "__main__":
    main()

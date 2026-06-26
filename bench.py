"""Deterministic benchmark across all kernels (single + multi-statement).

One strong, legal schedule per kernel; reports baseline + measured speedup
(clang -O3 + OpenMP) and the geometric mean. No LLM. Run: python3 bench.py
"""
import math

from compilot.backend_isl import environment
from compilot.kernels import REGISTRY, MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment

# single-statement schedules
SINGLE = {
    "gemm":  "reorder(i, k, j)\ntile2d(i, j, 64, 64)\nparallel(i_t)",
    "syrk":  "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "syr2k": "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "syrk_tri":  "tile2d(i, j, 32, 32)\nparallel(i_t)",
    "syr2k_tri": "tile2d(i, j, 32, 32)\nparallel(i_t)",
    "floydwarshall": "tile2d(i, j, 64, 64)",
}
# multi-statement schedules (one per statement; parallelize a non-reduction outer loop)
MULTI = {
    "2mm":  ["reorder(i,k,j)\ntile2d(i,j,64,64)\nparallel(i_t)",
             "reorder(i,l,j)\ntile2d(i,j,64,64)\nparallel(i_t)"],
    "3mm":  ["reorder(i,k,j)\nparallel(i)"] * 3,
    "mvt":  ["parallel(i)", "parallel(i)"],
    "atax": ["parallel(i)", ""],
    "bicg": ["", "parallel(i)"],
    "gesummv": ["parallel(i)", "parallel(i)"],
}


def main():
    print(f"{'kernel':16}{'baseline(s)':>12}{'speedup':>10}   type")
    print("-" * 60)
    speeds = []
    for name in SINGLE:
        env = environment(name)
        r = env.evaluate(SINGLE[name])
        if r.status == "success":
            speeds.append(r.speedup)
            print(f"{name:16}{env.baseline()['time']:12.4f}{r.speedup:9.2f}x   single")
    for name in MULTI:
        env = MultiEnvironment(MULTI_REGISTRY[name]())
        r = env.evaluate(MULTI[name])
        if r["status"] == "success":
            speeds.append(r["speedup"])
            print(f"{name:16}{env.baseline()['time']:12.4f}{r['speedup']:9.2f}x   multi")
    gm = math.exp(sum(math.log(x) for x in speeds) / len(speeds))
    print("-" * 60)
    print(f"{'GEOMEAN':16}{'':12}{gm:9.2f}x   ({len(speeds)} kernels)")


if __name__ == "__main__":
    main()

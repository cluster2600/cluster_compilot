"""Deterministic benchmark across all kernels (single + multi-statement).

One strong, legal schedule per kernel; reports baseline + measured speedup
(clang -O3 + OpenMP) and the geometric mean. No LLM. Run: python3 bench.py

Each speedup is env.confirm(interleaved=True): the median ratio of CONFIRM_ROUNDS
alternating baseline/candidate runs, not one run (single parallel runs vary 2-3x).
"""
import math

from compilot.backend_isl import environment
from compilot.kernels import REGISTRY, MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment

# single-statement schedules
SINGLE = {
    "gemm":  "reorder(i, k, j)\nparallel(i)",   # at N=512 the nest is cache-resident: 512-way rows beat 8 tiles
    "syrk":  "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "syr2k": "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "syrk_tri":  "tile2d(i, j, 32, 32)\nparallel(i_t)",
    "syr2k_tri": "tile2d(i, j, 32, 32)\nparallel(i_t)",
    "floydwarshall": "tile2d(i, j, 64, 64)",
}
# multi-statement schedules (one per statement; parallelize a non-reduction outer loop)
MULTI = {
    "2mm":  ["reorder(i,k,j)\nparallel(i)", "reorder(i,l,j)\nparallel(i)"],
    "3mm":  ["reorder(i,k,j)\nparallel(i)"] * 3,
    "mvt":  ["parallel(i)", "parallel(i)"],
    "atax": ["parallel(i)", ""],
    "bicg": ["", "parallel(i)"],
    "gesummv": ["parallel(i)", "parallel(i)"],
    "gemver": ["parallel(i)", "parallel(i)", "parallel(i)", "parallel(i)"],
    "covariance": ["parallel(j)", "parallel(i)", "tile2d(i,j,32,32)\nparallel(i_t)"],
    "doitgen": ["parallel(r)", "parallel(r)"],
}


def main():
    print(f"{'kernel':16}{'baseline(s)':>12}{'speedup':>10}   type")
    print("-" * 60)
    speeds = []
    for name in SINGLE:
        env = environment(name)
        sp = env.confirm(SINGLE[name], interleaved=True)
        if sp:
            speeds.append(sp)
            print(f"{name:16}{env.baseline()['time']:12.4f}{sp:9.2f}x   single")
    for name in MULTI:
        env = MultiEnvironment(MULTI_REGISTRY[name]())
        sp = env.confirm(MULTI[name], interleaved=True)
        if sp:
            speeds.append(sp)
            print(f"{name:16}{env.baseline()['time']:12.4f}{sp:9.2f}x   multi")
    gm = math.exp(sum(math.log(x) for x in speeds) / len(speeds))
    print("-" * 60)
    print(f"{'GEOMEAN':16}{'':12}{gm:9.2f}x   ({len(speeds)} kernels)")


if __name__ == "__main__":
    main()

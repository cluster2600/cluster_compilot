"""Deterministic benchmark: one strong, legal schedule per kernel.

Reports baseline time + measured speedup (clang -O3 + OpenMP) and the geometric
mean. No LLM — reproducible. Run: python3 bench.py
"""
import math

from compilot.backend_isl import environment

BENCH = {
    "gemm":  "reorder(i, k, j)\ntile2d(i, j, 64, 64)\nparallel(i_t)",
    "syrk":  "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "syr2k": "tile2d(i, j, 64, 64)\nparallel(i_t)",
    "floydwarshall": "tile2d(i, j, 64, 64)",
}


def main():
    print(f"{'kernel':16}{'baseline(s)':>12}{'speedup':>10}   schedule")
    print("-" * 70)
    speeds = []
    for name, sched in BENCH.items():
        env = environment(name)
        base = env.baseline()["time"]
        r = env.evaluate(sched)
        head = sched.splitlines()[0] + (" …" if "\n" in sched else "")
        if r.status == "success":
            speeds.append(r.speedup)
            print(f"{name:16}{base:12.4f}{r.speedup:9.2f}x   {head}")
        else:
            print(f"{name:16}{base:12.4f}{'[' + r.status + ']':>10}")
    gm = math.exp(sum(math.log(x) for x in speeds) / len(speeds))
    print("-" * 70)
    print(f"{'GEOMEAN':16}{'':12}{gm:9.2f}x   ({len(speeds)} kernels)")


if __name__ == "__main__":
    main()

"""Baseline comparison: ComPilot vs naive auto-parallelization.

The paper compares ComPilot against the Pluto polyhedral optimizer. Pluto's
bundled piplib does not compile on this toolchain (Darwin-27 / clang-22 rejects
its legacy K&R C; the LLVM-14 clang++ can't link C++), so we report a tractable
*proxy* baseline instead: NAIVE = parallelize the outermost non-reduction loop
only (a simple compiler heuristic), with no tiling/reorder/unroll. The ratio
shows how much ComPilot's polyhedral-guided schedule beats naive parallelism.

Run: python3 baselines.py [--size LARGE] [--rounds 5] [--kernels gemm,2mm]
"""
import math

from compilot.backend_isl import environment
from compilot.kernels import sized_kernel
from compilot.multikernel import MultiEnvironment
from bench import SINGLE, MULTI, args_parser, header, measure, selected   # ComPilot's strong schedules

# NAIVE: parallelize the outermost non-reduction loop only
NAIVE_SINGLE = {"gemm": "parallel(i)", "syrk": "parallel(i)",
                "syr2k": "parallel(i)", "floydwarshall": ""}
NAIVE_MULTI = {"2mm": ["parallel(i)", "parallel(i)"], "3mm": ["parallel(i)"] * 3,
               "mvt": ["parallel(i)", "parallel(i)"], "atax": ["parallel(i)", ""],
               "bicg": ["", "parallel(i)"], "gesummv": ["parallel(i)", "parallel(i)"]}


def sp_single(name, sched, args):
    if not sched:
        return 1.0
    base, sp = measure(lambda: environment(name, args.size), sched, args)
    return sp if base else None                  # None: skipped (e.g. baseline over --timeout)


def sp_multi(name, scheds, args):
    base, sp = measure(lambda: MultiEnvironment(sized_kernel(name, args.size)), scheds, args)
    return sp if base else None


def main():
    args = args_parser(__doc__).parse_args()
    header(args)
    print(f"{'kernel':16}{'naive':>9}{'ComPilot':>11}{'ratio':>9}")
    print("-" * 46)
    naive_all, comp_all = [], []
    # only the kernels with a defined NAIVE baseline (a subset of bench's SINGLE/MULTI)
    singles, multis = selected(args, NAIVE_SINGLE, NAIVE_MULTI)
    rows = [(n, sp_single(n, NAIVE_SINGLE[n], args), sp_single(n, SINGLE[n], args)) for n in singles] + \
           [(n, sp_multi(n, NAIVE_MULTI[n], args), sp_multi(n, MULTI[n], args)) for n in multis]
    for name, n, c in rows:
        if n is None or c is None:
            print(f"{name:16}  skipped (a run failed or exceeded --timeout)")
            continue
        naive_all.append(n); comp_all.append(c)
        print(f"{name:16}{n:8.2f}x{c:10.2f}x{c/n:8.2f}x", flush=True)
    if not comp_all:
        raise SystemExit("no kernel produced a speedup")
    gm = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs))
    print("-" * 46)
    print(f"{'GEOMEAN':16}{gm(naive_all):8.2f}x{gm(comp_all):10.2f}x{gm(comp_all)/gm(naive_all):8.2f}x")
    print(f"\nComPilot is {gm(comp_all)/gm(naive_all):.2f}x faster than naive auto-parallelization (geomean).")


if __name__ == "__main__":
    main()

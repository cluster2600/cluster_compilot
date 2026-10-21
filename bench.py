"""Deterministic benchmark across all kernels (single + multi-statement).

One strong, legal schedule per kernel; reports baseline + measured speedup
(clang -O3 + OpenMP) and the geometric mean. No LLM.

    python3 bench.py                         # PolyBench LARGE datasets (minutes)
    python3 bench.py --size MEDIUM --rounds 3
    python3 bench.py --kernels gemm,2mm

Each speedup is env.confirm(interleaved=True): the median ratio of `--rounds`
alternating baseline/candidate runs, not one run (single parallel runs vary 2-3x).
The default is PolyBench's LARGE datasets: at DEFAULT/MEDIUM sizes most nests fit
in cache, so locality transforms (tiling) can't show up in the numbers.
"""
import argparse
import math
import os

from compilot.backend_isl import environment
from compilot.kernels import SIZE_NAMES, sized_kernel
from compilot.multikernel import MultiEnvironment
from compilot import runner
from compilot.runner import CONFIRM_ROUNDS, busy_warning

# single-statement schedules
SINGLE = {
    "gemm":  "tile3d(i, j, k, 32, 128, 64)\nreorder(i_t, j_t, k_t, i, k, j)\nparallel(i_t)",
    "syrk":  "tile2d(i, j, 32, 32)\nparallel(i_t)",
    "syr2k": "tile2d(i, j, 32, 32)\nparallel(i_t)",
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


def args_parser(doc):
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--size", default="LARGE", choices=SIZE_NAMES,
                    help="PolyBench dataset class (default LARGE) or DEFAULT (built-in sizes)")
    ap.add_argument("--rounds", type=int, default=CONFIRM_ROUNDS,
                    help="alternating baseline/candidate run pairs per kernel")
    ap.add_argument("--kernels", default="", help="comma-separated subset (default: all)")
    ap.add_argument("--timeout", type=int, default=runner.RUN_TIMEOUT,
                    help="seconds per compile / run (naive LARGE baselines can take minutes)")
    return ap


def header(args):
    runner.RUN_TIMEOUT = args.timeout
    threads = os.environ.get("OMP_NUM_THREADS", f"default ({os.cpu_count()} CPUs)")
    print(f"size={args.size}  rounds={args.rounds}  OMP threads={threads}")
    if (warn := busy_warning()):
        print(f"warning: {warn}")
    print()


def selected(args, *groups):
    """The kernels of `groups` (dicts in run order) that --kernels asks for. Unknown
    names are an error, so a typo can't silently produce an empty benchmark."""
    want = [k.strip() for k in args.kernels.split(",") if k.strip()]
    known = [n for g in groups for n in g]
    unknown = sorted(set(want) - set(known))
    if unknown:
        raise SystemExit(f"--kernels: unknown or unbenchmarked {unknown}; choose from {known}")
    return [[n for n in g if not want or n in want] for g in groups]


def measure(make_env, sched, args):
    """(baseline seconds, confirmed speedup) or (None, reason) — one kernel failing
    (e.g. a naive LARGE baseline over --timeout) mustn't abort the whole benchmark."""
    try:
        env = make_env()
        sp = env.confirm(sched, rounds=args.rounds, interleaved=True)
        return (env.baseline()["time"], sp) if sp else (None, "candidate failed")
    except RuntimeError as e:                 # baseline failed (timeout, compile error)
        return None, str(e)[:70]


def main():
    args = args_parser(__doc__).parse_args()
    header(args)
    print(f"{'kernel':16}{'baseline(s)':>12}{'speedup':>10}   type")
    print("-" * 60)
    speeds = []
    singles, multis = selected(args, SINGLE, MULTI)
    jobs = [(n, "single", lambda n=n: environment(n, args.size), SINGLE[n]) for n in singles] + \
           [(n, "multi", lambda n=n: MultiEnvironment(sized_kernel(n, args.size)), MULTI[n]) for n in multis]
    for name, kind, make_env, sched in jobs:
        base, sp = measure(make_env, sched, args)
        if base is None:
            print(f"{name:16}{'—':>12}{'—':>10}   {kind}: skipped ({sp})", flush=True)
            continue
        speeds.append(sp)
        print(f"{name:16}{base:12.4f}{sp:9.2f}x   {kind}", flush=True)
    if not speeds:
        raise SystemExit("no kernel produced a speedup")
    gm = math.exp(sum(math.log(x) for x in speeds) / len(speeds))
    print("-" * 60)
    print(f"{'GEOMEAN':16}{'':12}{gm:9.2f}x   ({len(speeds)} kernels)")


if __name__ == "__main__":
    main()

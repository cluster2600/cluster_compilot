"""Evaluate ComPilot across kernels: per-kernel best speedup + geometric mean.

  python3 eval.py --mock                       # all kernels, scripted
  python3 eval.py --kernels gemm,syrk --k 3    # live Gemini, best-of-3
"""
import argparse
import math

from compilot.kernels import REGISTRY
from compilot.backend_isl import environment
from compilot.agent import best_of_k, run_dialogue
from compilot.llm import GeminiClient, MockClient


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--kernels", default=",".join(REGISTRY))
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    make = (lambda: MockClient()) if args.mock else (lambda: GeminiClient(model=args.model))
    names = [n.strip() for n in args.kernels.split(",") if n.strip()]
    driver = "mock" if args.mock else args.model
    print(f"driver={driver}  K={args.k}  iters={args.iters}  kernels={names}\n")

    results = {}
    for name in names:
        env = environment(name)
        print(f"### {name}  (baseline {env.baseline()['time']:.4f}s)")
        if args.k > 1:
            (sp, sched), _ = best_of_k(env, make, K=args.k, max_iters=args.iters, verbose=True)
        else:
            sp, sched, _ = run_dialogue(env, make(), max_iters=args.iters, verbose=True)
        results[name] = (sp, sched)
        print(f"  => {name}: {sp:.2f}x\n")

    print("=" * 48)
    speeds = [sp for sp, _ in results.values()]
    for name, (sp, _) in results.items():
        print(f"  {name:10} {sp:7.2f}x")
    print(f"  {'GEOMEAN':10} {geomean(speeds):7.2f}x   (ComPilot{'_%d' % args.k if args.k > 1 else ''}@{args.iters})")


if __name__ == "__main__":
    main()

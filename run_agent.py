"""Run the ComPilot agent on a kernel.

  python3 run_agent.py --mock                 # scripted, no API key
  python3 run_agent.py --iters 20             # live Gemini (key from OpenBao)
  python3 run_agent.py --k 5 --iters 20       # best-of-5
"""
import argparse

from compilot.backend_isl import environment
from compilot.agent import run_dialogue, best_of_k
from compilot.llm import GeminiClient, MockClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--kernel", default="gemm")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    env = environment(args.kernel)
    make = (lambda: MockClient()) if args.mock else (lambda: GeminiClient(model=args.model))
    print(f"kernel={args.kernel}  baseline={env.baseline()['time']:.4f}s  "
          f"driver={'mock' if args.mock else args.model}  K={args.k} iters={args.iters}\n")

    if args.k > 1:
        (sp, sched), runs = best_of_k(env, make, K=args.k, max_iters=args.iters)
        print(f"\nbest-of-{args.k}: {sp:.2f}x")
    else:
        sp, sched, _ = run_dialogue(env, make(), max_iters=args.iters)
    print(f"\n=== BEST {sp:.2f}x ===\n{sched}")


if __name__ == "__main__":
    main()

"""Run the ComPilot agent on a kernel.

  python3 run_agent.py --mock                 # scripted, no API key
  python3 run_agent.py --iters 20             # live Gemini (key from OpenBao)
  python3 run_agent.py --k 5 --iters 20       # best-of-5 (runs in parallel)
  python3 run_agent.py --k 5 --candidates 4   # + 4 parallel candidate schedules/turn

Requires Python 3.14+. Best-of-k runs and per-turn candidate schedules are
evaluated concurrently (threads): the LLM calls and the clang compile/run both
release the GIL, so this scales on the standard interpreter.
"""
import argparse
import sys

from compilot.backend_isl import environment
from compilot.agent import run_dialogue, run_dialogue_multi, best_of_k, run_dialogue_moa
from compilot.llm import GeminiClient, OpenAIClient, MockClient
from compilot.kernels import MULTI_REGISTRY, STENCIL_REGISTRY
from compilot.multikernel import MultiEnvironment
from compilot.stencil import StencilEnvironment


def make_client(spec, base_url, temperature=0.7):
    """spec is 'backend:model' — gemini:gemini-2.5-pro, local:qwen2.5-coder:32b, mock.
    Split on the first ':' only, so model names with colons (Ollama tags) survive."""
    backend, _, model = spec.partition(":")
    if backend == "mock":
        return MockClient()
    if backend == "local":
        return OpenAIClient(model=model, base_url=base_url, temperature=temperature)
    if backend == "gemini":
        return GeminiClient(model=model, temperature=temperature)
    raise SystemExit(f"unknown backend in {spec!r} (use gemini:…, local:…, or mock)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--kernel", default="gemm")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--candidates", type=int, default=3,
                    help="schedules proposed and evaluated in parallel per turn (1 = classic)")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--backend", default="gemini", choices=["gemini", "local"],
                    help="provider for the single-model path (local = OpenAI-compatible server)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible endpoint for --backend local / local: MoA specs")
    ap.add_argument("--moa", default="",
                    help="enable Mixture-of-Agents: comma-separated reference specs, "
                         "e.g. 'gemini:gemini-2.5-flash,local:qwen2.5-coder:32b'")
    ap.add_argument("--aggregator", default="",
                    help="MoA aggregator spec (default: the single-model --backend:--model)")
    args = ap.parse_args()

    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    runtime = f"py{sys.version_info.major}.{sys.version_info.minor} {'GIL' if gil else 'free-threaded'}"

    base_spec = "mock" if args.mock else f"{args.backend}:{args.model}"
    make = lambda: make_client(base_spec, args.base_url)

    if args.kernel in MULTI_REGISTRY or args.kernel in STENCIL_REGISTRY:   # >1 statement
        if args.kernel in STENCIL_REGISTRY:
            menv = StencilEnvironment(STENCIL_REGISTRY[args.kernel]())
            kind = "stencil"
        else:
            menv = MultiEnvironment(MULTI_REGISTRY[args.kernel]())
            kind = "multi-statement"
        print(f"kernel={args.kernel} ({kind}, {len(menv.mk.statements)} stmts)  "
              f"baseline={menv.baseline()['time']:.4f}s  driver={'mock' if args.mock else args.model}\n")
        sp, best = run_dialogue_multi(menv, make(), max_iters=args.iters)
        print(f"\n=== BEST {sp:.2f}x ===")
        for i, s in enumerate(best or []):
            print(f"  [stmt {i}] {s.strip() or '(identity)'}")
        return

    env = environment(args.kernel)
    print(f"kernel={args.kernel}  baseline={env.baseline()['time']:.4f}s  "
          f"driver={'mock' if args.mock else args.model}  K={args.k} iters={args.iters} "
          f"candidates={args.candidates}  [{runtime}]\n")

    if args.moa:
        refs = [make_client(s.strip(), args.base_url, temperature=0.9)
                for s in args.moa.split(",") if s.strip()]
        agg_spec = args.aggregator or base_spec
        agg = make_client(agg_spec, args.base_url, temperature=0.4)
        print(f"MoA: {len(refs)} references [{args.moa}] -> aggregator [{agg_spec}]\n")
        sp, sched, _ = run_dialogue_moa(env, refs, agg, max_iters=args.iters,
                                        candidates_per_turn=args.candidates)
    elif args.k > 1:
        (sp, sched), runs = best_of_k(env, make, K=args.k, max_iters=args.iters,
                                      candidates_per_turn=args.candidates)
        print(f"\nbest-of-{args.k}: {sp:.2f}x")
    else:
        sp, sched, _ = run_dialogue(env, make(), max_iters=args.iters,
                                    candidates_per_turn=args.candidates)
    print(f"\n=== BEST {sp:.2f}x ===\n{sched}")


if __name__ == "__main__":
    main()

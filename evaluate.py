"""Full evaluation harness (paper RQ1 / RQ2 / RQ9).

For each kernel we run a POOL of independent agent dialogues, each capped at T
iterations, and record (best legal speedup, tokens). From the pool we report:

  ComPilot@T     median single-run speedup        (RQ1)
  ComPilot_K@T   typical best-of-K speedup         (RQ9)  median over resampled K-maxes
  geomean        geometric mean across kernels
  95% CI         bootstrap confidence interval on each statistic
  tokens / cost  cumulative LLM token usage + estimated $ (RQ2)

  python3 evaluate.py --mock --pool 8                 # deterministic-ish, no key
  python3 evaluate.py --kernels gemm,syrk --pool 10 --k 5 --iters 20   # live Gemini
"""
import argparse
import math
import random
import statistics as stats

from compilot.backend_isl import environment
from compilot.agent import run_dialogue, run_dialogue_multi
from compilot.llm import GeminiClient, MockClient
from compilot.kernels import MULTI_REGISTRY
from compilot.multikernel import MultiEnvironment

# gemini-2.5-flash list price (USD per 1M tokens); override with --price-in/--price-out
PRICE_IN, PRICE_OUT = 0.30, 2.50


def geomean(xs):
    return math.exp(sum(math.log(max(x, 1e-9)) for x in xs) / len(xs)) if xs else 0.0


def _bootstrap(samples, stat, B=2000, seed=12345):
    """Bootstrap 95% CI for `stat` over `samples`. Reproducible (fixed-seed RNG)."""
    n = len(samples)
    if n < 2:
        return (samples[0], samples[0]) if samples else (0.0, 0.0)
    rng = random.Random(seed)
    reps = sorted(stat([rng.choice(samples) for _ in range(n)]) for _ in range(B))
    return reps[int(0.025 * B)], reps[int(0.975 * B)]


def best_of_k_estimate(pool, k, seed=777):
    """Median of max-over-k resamples (typical best-of-K), per the paper."""
    if not pool:
        return 0.0
    rng = random.Random(seed)
    maxes = [max(rng.choice(pool) for _ in range(k)) for _ in range(1000)]
    return stats.median(maxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--kernels", default="gemm,syrk,syr2k,floydwarshall,2mm,3mm,mvt,atax,bicg,gesummv")
    ap.add_argument("--pool", type=int, default=8, help="independent runs per kernel")
    ap.add_argument("--k", type=int, default=5, help="K for best-of-K")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--price-in", type=float, default=PRICE_IN)
    ap.add_argument("--price-out", type=float, default=PRICE_OUT)
    args = ap.parse_args()

    names = [n.strip() for n in args.kernels.split(",") if n.strip()]
    tok_in = tok_out = 0
    per_kernel = {}

    for name in names:
        is_multi = name in MULTI_REGISTRY
        env = MultiEnvironment(MULTI_REGISTRY[name]()) if is_multi else environment(name)
        pool = []
        for r in range(args.pool):
            llm = MockClient() if args.mock else GeminiClient(model=args.model)
            if is_multi:
                sp, _ = run_dialogue_multi(env, llm, max_iters=args.iters, verbose=False)
            else:
                sp, _, _ = run_dialogue(env, llm, max_iters=args.iters, verbose=False)
            pool.append(sp)
            tok_in += getattr(llm, "in_tokens", 0)
            tok_out += getattr(llm, "out_tokens", 0)
        at_t = stats.median(pool)
        k_at_t = best_of_k_estimate(pool, args.k)
        ci = _bootstrap(pool, stats.median)
        per_kernel[name] = (at_t, k_at_t, ci, pool)
        print(f"{name:14} ComPilot@{args.iters}={at_t:6.2f}x  "
              f"ComPilot_{args.k}@{args.iters}={k_at_t:6.2f}x  "
              f"95%CI=[{ci[0]:.2f}, {ci[1]:.2f}]  (pool={args.pool})")

    print("=" * 72)
    gm_at = geomean([v[0] for v in per_kernel.values()])
    gm_k = geomean([v[1] for v in per_kernel.values()])
    gm_ci = _bootstrap([v[0] for v in per_kernel.values()], geomean)
    print(f"GEOMEAN  ComPilot@{args.iters}={gm_at:.2f}x  "
          f"ComPilot_{args.k}@{args.iters}={gm_k:.2f}x  95%CI=[{gm_ci[0]:.2f}, {gm_ci[1]:.2f}]")

    cost = tok_in / 1e6 * args.price_in + tok_out / 1e6 * args.price_out
    print(f"\nRQ2 tokens: in={tok_in:,} out={tok_out:,}  est. cost ${cost:.4f} "
          f"({'mock — no real tokens' if args.mock else args.model})")


if __name__ == "__main__":
    main()

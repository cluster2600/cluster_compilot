# cluster_compilot

A faithful, from-scratch implementation of **ComPilot** — *Agentic Auto-Scheduling: LLM-Guided Loop Optimization* ([arXiv:2511.00592](https://arxiv.org/abs/2511.00592), Merouani, Kara Bernou & Baghdadi, PACT 2025).

An off-the-shelf LLM acts as an agent that proposes loop transformations. A compiler-grade **polyhedral legality engine** proves whether each schedule is legal, and the transformed code is compiled and **executed for real wall-clock speedup**. The LLM iterates on that feedback. No fine-tuning.

> **Status:** runs live. Gemini 2.5-flash + ISL legality + clang execution reach **42× on GEMM** end-to-end. We also build the **real Tiramisu compiler** and cross-validate our legality against it (**4/4**).

![architecture](docs/images/architecture.png)

## Documentation

Full docs live in [`docs/`](docs/):

| Doc | What's in it |
|---|---|
| [Architecture](docs/architecture.md) | The four diagrams (rendered + live mermaid): system, dialogue, legality, backends |
| [How it works](docs/how-it-works.md) | Evaluation pipeline (parse → θ′ → ISL legality → measure), feedback categories, the DSL |
| [Building (step by step)](docs/building.md) | Prereqs → deps → key → smoke test → optional Tiramisu build |
| [User guide (step by step)](docs/user-guide.md) | Run the agent, eval, benchmark, tests; write schedules; add a kernel |
| [Test results](docs/test-results.md) | Real test + benchmark output, with screenshots |

## Quick start

```bash
git clone https://github.com/cluster2600/cluster_compilot.git
cd cluster_compilot
pip install -r requirements.txt        # islpy, certifi
brew install libomp                    # OpenMP for clang (macOS)

python3 -m tests.test_legality         # prove the legality oracle (10/10)
python3 run_agent.py --mock            # full agent loop, no API key
python3 run_agent.py --iters 15        # live Gemini (key from env or OpenBao)
```

See the [building guide](docs/building.md) and [user guide](docs/user-guide.md) for everything else (live keys, Tiramisu build, adding kernels).

## Test results

All suites pass; the benchmark is reproducible (`python3 bench.py`). Full output + screenshots in [docs/test-results.md](docs/test-results.md).

![test suite](docs/images/shot_tests.png)

| Suite | Result |
|---|---|
| `test_legality` | **10/10** — accepts interchange/tile/skew; rejects `reverse(k)`; flags `parallel(k)` |
| `test_environment` | GEMM `reorder` ~7.7×, `tile2d+parallel` ~11–14×; illegal schedules rejected pre-execution |
| `test_multistatement` | **3/3** — fused/distributed legal, reordered illegal |
| `test_tiramisu_parity` | **4/4** — our ISL engine agrees with the real Tiramisu compiler |

**Benchmark (deterministic):** gemm ~26–31× · syrk ~5× · syr2k ~5× · floyd ~1× (correctly un-parallelizable) → **geomean ~5×**. Live Gemini: **42× GEMM**, ~13× geomean.

## Repo layout

| File | Role |
|---|---|
| `compilot/kernels.py` | kernels as schedulable loop nests (exec + poly specs, `REGISTRY`) |
| `compilot/schedule.py` | parse the 9-primitive schedule DSL |
| `compilot/scheduler.py` | transforms → ISL schedule map θ′ |
| `compilot/polyhedral.py` | **ISL legality + parallelism oracle** |
| `compilot/polyhedral_multi.py` | multi-statement **legality** (2d+1 schedules) |
| `compilot/multikernel.py` | multi-statement **execution** (sequence of statements, e.g. 2mm) |
| `compilot/codegen.py` · `runner.py` | emit timed C · compile (`clang -O3 +OpenMP`) + run |
| `compilot/backend_isl.py` | `Environment`: legality → codegen → measured speedup |
| `compilot/backends/tiramisu.py` | drive the **real** libtiramisu legality |
| `compilot/prompt.py` · `feedback.py` · `llm.py` · `secrets.py` · `agent.py` | context prompt · 5 feedback categories · Gemini client · OpenBao · dialogue + best-of-K |
| `run_agent.py` · `eval.py` · `bench.py` · `evaluate.py` | run the agent · geomean eval · benchmark · **full metrics (ComPilot@T, CIs, cost)** |
| `tests/` | legality (10/10), environment, multi-statement (3/3), multi-kernel (10 PolyBench), fusion, Tiramisu parity (4/4) |
| `third_party/tiramisu/` | exact Tiramisu backend — **built** (`libtiramisu.dylib`); gitignored |

**Kernels (14):** single — `gemm`, `syrk`, `syr2k`, `syrk_tri`, `syr2k_tri`, `floydwarshall`; multi-statement — `2mm`, `3mm`, `mvt`, `atax`, `bicg`, `gesummv`, `gemver`, `covariance`. Spanning matmul, matvec, reduction, **triangular** domains, **fusion**, **in-place** (reset zero/reinit/none), and **datamining**.

## Status & roadmap

**Working & live:** ISL legality oracle + parallelism check; **all 9 primitives execute** (incl. `skew`, `reverse`, `fuse`); clang/OpenMP execution; full agent dialogue with Gemini via OpenBao for **both single- and multi-statement** kernels; **10 kernels** (4 single + 6 multi: 2mm/3mm/mvt/atax/bicg/gesummv); exact Tiramisu backend **built** and driven (ISL↔Tiramisu **4/4**); multi-statement legality (3/3) + execution (2mm live **28.6×**); loop **fusion** (gesummv 1.7×); evaluation harness — **ComPilot@T, ComPilot_K@T, bootstrap 95% CIs, token/cost (RQ1/RQ2/RQ9)**.

**Baseline:** `baselines.py` — **ComPilot vs naive auto-parallelization**: geomean **6.17× vs 3.31× → 1.86× faster** (matmul kernels 4.6–6.2× faster; matvec memory-bound, ~1.1–1.35×). Polyhedral **Pluto** itself does **not build** on this toolchain (Darwin-27/clang-22 rejects its bundled piplib's legacy K&R C — `conflicting types`/`unknown type name`; the LLVM-14 clang++ can't link C++), so naive auto-parallel is the proxy comparison.

**Pending:**
- **Tiramisu execution timing** — Tiramisu Halide **codegen works** (lowers a scheduled GEMM to a ~128 KB object); the remaining piece is a Halide-buffer wrapper to *run+time* that object (clang already measures speedup)
- **full PolyBench/C 4.2.1 (150 instances)** — **14/30 kernels** done (matmul/matvec/triangular/multi-statement/fusion/in-place/datamining all working). The rest need: time-loop support (stencils — jacobi/seidel/heat/fdtd/adi), loop-carried solvers (cholesky/lu/trisolv/durbin), 3-D arrays (doitgen); plus the ×5 size classes

## Reference

Merouani, Kara Bernou, Baghdadi. *Agentic Auto-Scheduling: An Experimental Study of LLM-Guided Loop Optimization.* PACT 2025. arXiv:2511.00592.

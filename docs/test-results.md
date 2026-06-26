# Test results

All suites pass and the benchmark is reproducible. Captured on a multi-core macOS / Apple-Silicon machine — **speedups are machine-dependent; legality verdicts are not.** Regenerate everything with the commands shown.

## Test suite

```bash
for t in legality environment multistatement tiramisu_parity; do python3 -m tests.test_$t; done
```

![test suite](images/shot_tests.png)

| Suite | Result |
|---|---|
| `test_legality` — ISL oracle distinguishes legal vs illegal | **10/10** — accepts interchange/tile/skew; rejects `reverse(k)`; flags `parallel(k)` non-parallelizable |
| `test_environment` — legality + real measured speedup on GEMM | baseline ≈0.17 s; `reorder(i,k,j)` ≈7.7×; `tile2d+parallel` ≈11–14×; `reverse(k)`→illegal; `parallel(k)`→parallel_illegal |
| `test_multistatement` — producer→consumer legality | **3/3** — fused legal, distributed legal, reordered (consumer-first) illegal |
| `test_tiramisu_parity` — ISL vs the **real Tiramisu compiler** | **4/4 agree** — interchange, `reverse(k)`=illegal, `reverse(i)`, tile2d |

## Benchmark

One strong, legal schedule per kernel — deterministic, no LLM (`clang -O3 + OpenMP`):

```bash
python3 bench.py
```

![benchmark](images/shot_benchmark.png)

| Kernel | Baseline | Speedup | Schedule |
|---|---|---|---|
| gemm | ~0.17 s | **~26–31×** | `reorder(i,k,j)` + `tile2d(64,64)` + `parallel(i_t)` |
| syrk | ~0.10 s | **~5×** | `tile2d(64,64)` + `parallel(i_t)` |
| syr2k | ~0.11 s | **~5×** | `tile2d(64,64)` + `parallel(i_t)` |
| floydwarshall | ~0.07 s | **~1×** | `tile2d(64,64)` — *cannot* parallelize under sound legality |
| **geomean** | | **~5×** | |

Floyd-Warshall at ~1× is the honest, *correct* result: row/column `k` is written and read across iterations, so sound polyhedral analysis forbids parallelizing `i`/`j` (only the non-negative-diagonal *semantics* would allow it, which a syntactic checker cannot assume — Pluto is conservative here too).

## Agent loop (live behaviour, mock driver shown)

```bash
python3 run_agent.py --mock --iters 8     # deterministic, no API key
```

![agent loop](images/shot_agent.png)

With **live Gemini** the agent tailors schedules per kernel and reaches **42× on GEMM** and a **~13× geomean** (`python3 run_agent.py --iters 15`, key from env or OpenBao).

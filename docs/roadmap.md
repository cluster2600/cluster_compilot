# Roadmap

Where the project goes after 0.3.0, in priority order. Each item says why it is there, which piece of evidence it rests on, and when it counts as done. Written 2026-10-21, against `main` at `362a080`.

The order follows one rule: **make the numbers trustworthy and comparable before adding features that are supposed to move them.** At the moment the Linux figures come from a busy host, and the live-LLM and Apple Silicon figures predate real tiling. The only baseline is naive auto-parallelization. Until that changes, there is no way to tell whether a new transform helps.

| # | Milestone | Item | Size | Waits on |
|---|---|---|---|---|
| 1 | 0.3.1 | Python 3.15 in CI | S | — |
| 2 | 0.3.1 | Remove `polyhedral_multi.py` | S | — |
| 3 | 0.3.1 | Publish to PyPI; npm via trusted publishing | M | PyPI and npmjs.com settings (owner) |
| 4 | 0.3.1 | Review the commits that never got a review pass | S | — |
| 5 | 0.4.0 | Polly baseline | M | — |
| 6 | 0.4.0 | Machine-readable benchmark output | S | — |
| 7 | 0.4.0 | Quiet-machine numbers (Apple Silicon) + Tiramisu parity | S | the Mac |
| 8 | 0.4.0 | Live LLM evaluation at 0.3.x | M | a Gemini key |
| 9 | 0.5.0 | Parallel reductions | L | 5, 6 |
| 10 | 0.5.0 | Tiling and interchange for imperfect nests | L | 5, 6 |
| 11 | later | Tiramisu parity in CI | L | — |
| 12 | later | Free-threaded 3.15t | S | islpy `cp315t` wheel |
| 13 | later | Benchmark history on a dedicated runner | M | 6, a runner |
| 14 | later | `vectorize` primitive | M | 5 (measure first) |
| 15 | later | Remaining research questions from the paper | L | 8 |

## 0.3.1: quick wins

**1. Python 3.15 in CI.** Python 3.15.0 was released on 2026-10-09, and islpy 2026.2.2 now ships a `cp315` wheel (`python3 check_py315.py` reports `READY`). The README's Python 3.15 section still calls islpy the blocker.
*Done when:* `tests.yml` runs on a 3.14 + 3.15 matrix and passes, and the README section says 3.15 is supported.

**2. Remove `polyhedral_multi.py`.** No production code imports it any more. Since 0.3.0, multi-statement and stencil kernels check each statement with `polyhedral.py` and fusion with `union.py`; imperfect nests use `union.py`. Only `tests/test_multistatement.py` and `tests/test_fusion.py` still use it, so the 3/3 multi-statement result tests a model that nothing runs. The README layout table and the `multikernel.py` docstring still describe it as the multi-statement legality engine.
*Done when:* both tests check the same cases through `union.py`, the module is deleted, and the docs no longer mention it.

**3. Publish to PyPI; npm via trusted publishing.** The npm package is only a launcher. Its README asks users to clone the repo and run `pip install -e .`, because the Python package is not on PyPI; the name `compilot` is still free there. Publishing to npm needs browser 2FA, and npm hides the approval link when its output isn't a terminal. With trusted publishing (OIDC from a GitHub Actions workflow) on both registries, a GitHub release publishes the image, the wheel and the npm package together.
*Done when:* publishing a GitHub release also publishes `compilot` to PyPI and `@cluster2600/compilot-mcp` to npm with no local login, and the npm README's prerequisite is `pip install compilot`.
*Waits on:* the owner adding the trusted publishers on pypi.org and npmjs.com.

**4. Review the unreviewed commits.** The last Codex pass on #33 never finished, so `db3f4a1` (loop depth from the transformed schedule) and `0e89429` (docs) were merged without a review.
*Done when:* those commits have had a review pass and any findings are fixed.

## 0.4.0: numbers that can be trusted and compared

**5. Polly baseline.** The README says Pluto does not build, so `baselines.py` compares against naive auto-parallelization instead. However, the Docker image's clang 19 already includes Polly (`/usr/lib/llvm-19/lib/LLVMPolly.so`), which is a real polyhedral optimizer. A quick probe in the image on a 1024³ gemm (busy host, two runs each):

| Build | Time |
|---|---|
| `clang -O3 -fopenmp` | 6.3–14.4 s |
| `+ -mllvm -polly -mllvm -polly-parallel` | 0.27–0.29 s |

ComPilot's gemm at LARGE measured 10.4× in the same environment, so Polly may well beat it on the dense kernels. That is exactly the comparison the project should publish.
*Done when:* `baselines.py` has a Polly column for every kernel at LARGE, taken with the same interleaved-median harness, and the README's baseline paragraph reports it.

**6. Machine-readable benchmark output.** `bench.py` and `baselines.py` print tables only. With a `--json` option, results from different hosts and versions could be diffed and kept in the repo (for example `docs/results/<version>-<host>.json`) instead of being copied into prose by hand.
*Done when:* both scripts write JSON with per-kernel medians, the spread, thread count, host load and dataset, and the README numbers are generated from those files.

**7. Quiet-machine numbers and Tiramisu parity.** Every Linux number since 0.3.0 comes from a WSL host that was busy at the time; there, even A/A runs of a single binary vary from 0.25× to 3.8×. The Apple Silicon figures in the README (gemm ~26–31×, geomean ~5×) predate real tiling and interleaved medians. The Tiramisu backend is built only on the Mac, and CI skips its test, so ISL↔Tiramisu parity (4/4) has not been re-checked since `tile2d`/`tile3d` changed meaning in 0.3.0.
*Done when:* `bench.py` and `baselines.py` (at LARGE) and `tests.test_tiramisu_parity` have been re-run on the Mac, and their output replaces the pre-0.3 numbers.
*Waits on:* the Mac.

**8. Live LLM evaluation at 0.3.x.** The headline "live Gemini 42× GEMM, ~13× geomean" was measured with strip-mining `tile2d` and single timed runs. The prompt now teaches the new tile semantics and `fuse`/`shift`, but no model has been run against it. The default model (`gemini-2.5-flash`) and the price constants in `evaluate.py` (0.30/2.50 USD per 1M tokens) also need checking against what Google serves and charges today.
*Done when:* `evaluate.py` (RQ1/RQ2/RQ9) has run at 0.3.x on a model that is currently served, and its results and cost replace the old headline.
*Waits on:* a key in `GEMINI_API_KEY`. Exporting it in the shell is enough; no OpenBao is needed.

## 0.5.0: more speedup the oracle can prove

**9. Parallel reductions.** `kernels.py` declares a `reduction` set on gemm, syrk, atax, bicg and others, but no code reads that field. A reduction loop can only run as plain `parallel()`, which the oracle correctly rejects. This is the remaining lever for the matvec kernels: atax's `y[j] +=` and bicg's `sv[j] +=` reduce over the outer loop, and today their schedules are identical in both baseline columns. A sound design treats only the declared reduction's dependences as relaxable at that loop and emits an OpenMP `reduction` clause. The checksum check then needs a floating-point tolerance, because reassociation changes rounding.
*Done when:* `parallel(L)` on a declared reduction loop is proven legal (every other dependence still checked), produces `reduction(+: …)` code, and passes the checksum within tolerance. The atax and bicg baselines should show a difference.

**10. Tiling and interchange for imperfect nests.** `imperfect.py` refuses every transform except `parallel()`. Yet `union.py` already provides exact legality and ISL codegen for the whole statement tree. The missing piece is mapping the DSL's tile and interchange onto per-statement schedules. This is where the solver speedups would come from: the README admits that the parallel-only solvers rarely beat serial `-O3`.
*Done when:* `tile2d` and `reorder` are accepted on lu, cholesky, trmm and symm when legal, generated by ISL, checksum-verified and measured against Polly (item 5).

## Later

**11. Tiramisu parity in CI.** libtiramisu needs LLVM and Halide built from source, which is why it exists only on the Mac. A separate image, built by hand or nightly with layer caching, would let parity run on Linux and catch drift between the two legality engines.

**12. Free-threaded 3.15t.** This waits on islpy publishing a `cp315t` wheel. `check_py315.py` already tracks it.

**13. Benchmark history on a dedicated runner.** Once item 6 exists, a self-hosted runner (the Mac) can run the benchmarks on each release and flag regressions beyond the measured noise band.

**14. `vectorize` primitive.** Tiramisu has one and the DSL does not, but clang already auto-vectorizes innermost loops. Item 5's per-kernel numbers should show whether an explicit `omp simd` primitive would buy anything before it is built.

**15. Remaining research questions from the paper.** `evaluate.py` implements RQ1, RQ2 and RQ9. Model comparisons and ablations, such as feedback categories and context, become worth running once item 8 gives a working live setup.

## Not planned

- **Floyd-Warshall `i`/`j` parallelism.** At step `k`, row and column `k` are rewritten with values that do not change, because `path[k][k] = 0`. The writes still happen, so value-based dependence analysis sees the same dependences that memory-based analysis does. Proving the loop parallel would take a semantic assertion, and that would break the rule that only proven schedules run.
- **Pluto.** Item 5's Polly baseline replaces it.

# How it works

**What ComPilot does.** An LLM is given a loop nest and its baseline runtime, then proposes a *schedule* (a sequence of loop transformations) inside `<schedule>…</schedule>` tags. The environment checks the schedule and returns one of five outcomes; the LLM uses that feedback to refine its next proposal. It keeps the best legal speedup and stops on a stop-token or an iteration cap. No fine-tuning — the intelligence is an off-the-shelf model; correctness comes from the compiler.

See the [architecture diagrams](architecture.md) for the visual version.

## How a proposal is evaluated

`backend_isl.Environment.evaluate(schedule)`:

1. **Parse** the schedule (`schedule.py`) into the 9-primitive DSL.
2. **Build θ′** (`scheduler.py`) — the new schedule as an ISL map from each iteration to its new logical time vector.
3. **Prove legality** (`polyhedral.py`): compute all memory dependences `D` (RAW/WAR/WAW, ordered by the original schedule); the schedule is legal iff every dependence stays lexicographically forward under θ′:
   `∀ (i→i′) ∈ D :  θ′(i) ≺ₗₑₓ θ′(i′)`.
   A loop level is parallelizable iff no dependence is *carried* at that level.
4. **Generate** (`codegen.isl_nest`): ISL's AST generator scans the iteration domain in the order θ′ defines and prints the loops; `parallel`/`unroll` become pragmas on the loops of their schedule dimension. The code that runs is generated from the *same* θ′ that was proven legal, so legality and execution can't drift apart. (The baseline is printed separately, straight from the kernel's C loop nest.)
5. **Measure** (`runner.py`): compile with `clang -O3 + OpenMP`, run, and report `baseline_time / new_time`. The output checksum is cross-checked against the baseline — a second, independent correctness guard that also catches a polyhedral spec that disagrees with the C body.

Single runs of a parallel kernel are noisy (the same schedule can measure 2–3× apart), so when a raw speedup would set a new best, the agent first **confirms** it (`env.confirm` → `runner.confirm_speedup`): the schedule is re-run `CONFIRM_ROUNDS` (5) times and the median is what's kept. Without this the search reports the luckiest of many samples. `bench.py` and `baselines.py` use the stricter `confirm(..., interleaved=True)`, which alternates baseline and candidate runs (ABBA…) and keeps the median ratio, so baseline noise and drift cancel too — at the cost of a baseline run per round.

## Feedback categories (`feedback.py`)

`success` (with speedup) · `illegal` (dependence violation) · `parallel_illegal` (loop carries a dependence) · `invalid` (unparseable, or names a loop that doesn't exist) · `compile/runtime_error`.

## Legality backends

- **`backend_isl`** uses ISL directly (`islpy`) — the same library Tiramisu wraps.
- **`backends/tiramisu.py`** drives the **real Tiramisu compiler** we built; the two agree **4/4** on directly-comparable transforms (see [test results](test-results.md)).
- **`union.py`** models several statements at once (2d+1 schedules, accesses read from the C): exact legality for imperfect nests and for fused statement sequences, plus the ISL codegen for both. (`polyhedral_multi.py` is the earlier 2d+1 legality prototype `test_multistatement` exercises.)

## The schedule DSL

```
interchange(La, Lb)              reorder(La, Lb, Lc, ...)
tile(L, T)                       tile2d(La, Lb, Ta, Tb)      tile3d(La, Lb, Lc, Ta, Tb, Tc)
parallel(L)                      unroll(L, F)
skew(Ltarget, Lsrc, factor)      reverse(L)
shift(L, c)                      fuse(L)                     (multi-statement / stencil)
```

Legality and execution cover all of these for single-statement nests and per statement in multi-statement/stencil kernels; `fuse` (and `shift` for alignment) run the statement sequence as one union. Imperfect-nest solvers execute `parallel` only.

Semantics worth knowing:
- **Tiling** follows Tiramisu: tile loops are inserted together, in argument order, just outside the outermost tiled loop. On nest `(i, k, j)`, `tile2d(i, j, 32, 32)` gives `(i_t, j_t, i, k, j)`; a single `tile(L, T)` is a strip-mine (`L_t` directly outside `L`).
- **`skew`/`reverse`** act on a loop's current value, so `skew(j, i, 1)` + `interchange(i, j)` is a wavefront (`j+i` outermost) and `reverse(i_t)` reverses whole tiles.
- **`fuse(L)`** in statement k's block runs it inside statement k−1's loops down to and including `L` (loops matched by position), after k−1's body there. **`shift(L, c)`** offsets `L` by `c`; alone it changes nothing, but it is what makes many fusions legal — jacobi's second sweep reads `B[i+1]`, so `shift(i, 1)` then `fuse(i)` delays it one iteration.
- Every loop label an op names is validated when θ′ is built; a typo is an `invalid` result, not an error.

## Secrets

The Gemini key is fetched from **OpenBao** at runtime (`secrets.py`) — never written to disk or printed.

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
4. **Measure** (`codegen.py` + `runner.py`): if legal, emit C, compile with `clang -O3 + OpenMP`, run, and report `baseline_time / new_time`. The output checksum is cross-checked against the baseline — a second, independent correctness guard.

## Feedback categories (`feedback.py`)

`success` (with speedup) · `illegal` (dependence violation) · `parallel_illegal` (loop carries a dependence) · `invalid` (unparseable) · `compile/runtime_error`.

## Legality backends

- **`backend_isl`** uses ISL directly (`islpy`) — the same library Tiramisu wraps.
- **`backends/tiramisu.py`** drives the **real Tiramisu compiler** we built; the two agree **4/4** on directly-comparable transforms (see [test results](test-results.md)).
- **`polyhedral_multi.py`** extends legality to multiple statements (2d+1 schedules) — the gate for `fuse`/`shift` and multi-statement kernels.

## The 9-primitive schedule DSL

```
interchange(La, Lb)              reorder(La, Lb, Lc, ...)
tile(L, T)                       tile2d(La, Lb, Ta, Tb)      tile3d(La, Lb, Lc, Ta, Tb, Tc)
parallel(L)                      unroll(L, F)
skew(Ltarget, Lsrc, factor)      reverse(L)
```

Legality is checked for all nine; execution currently covers the first seven (skew/reverse legality-checked; `fuse`/`shift` need the multi-statement model).

## Secrets

The Gemini key is fetched from **OpenBao** at runtime (`secrets.py`) — never written to disk or printed.

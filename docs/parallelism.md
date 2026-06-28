# Parallelism (Python 3.14)

ComPilot requires **Python 3.14+** and fans the search out across threads. The
guiding principle: **parallelize the search, serialize the measurement.** The
expensive work — Gemini calls and the `clang` compile/run — is I/O- and
subprocess-bound and releases the GIL, so threads scale on the *standard*
interpreter. Two locks keep the result correct and the numbers trustworthy.

> Free-threaded `3.14t` is **not** used: `islpy` ships no `cp314t` wheel, and the
> in-process polyhedral work is a small slice of wall-clock anyway, so removing
> the GIL would buy little here.

## Where the fan-out happens

| Axis | Flag | What runs concurrently |
|------|------|------------------------|
| **best-of-k** | `--k K` | the K independent dialogues (each its own LLM client) |
| **candidates / turn** | `--candidates N` | up to N schedules proposed in one turn |

```bash
python3 run_agent.py --k 5 --candidates 4 --iters 20
```

## Fan-out and the two locks

The agent fans out into many concurrent `env.evaluate()` calls. Each call passes
through a fast legality gate (serialized) and an expensive compile/measure stage
(compile parallel, timed run serialized).

![parallelism architecture](images/parallelism-arch.png)

<details><summary>mermaid source</summary>

```mermaid
flowchart TD
  subgraph POOL["ThreadPoolExecutor — best-of-k (parallel runs)"]
    R1["run 1 · dialogue"]
    R2["run 2 · dialogue"]
    Rk["run K · dialogue"]
  end
  subgraph TURN["each turn — up to N candidate schedules"]
    C1["candidate 1"]
    C2["candidate 2"]
    Cn["candidate N"]
  end
  R1 --> TURN
  C1 --> EV["env.evaluate()"]
  C2 --> EV
  Cn --> EV
  EV --> L1{"_ISL_LOCK<br/>legality: build_theta · is_legal · is_parallel<br/>(islpy DEFAULT_CONTEXT — fast, serialized)"}
  L1 -- illegal --> BAD["reject (no code run)"]
  L1 -- legal --> CG["codegen → clang -O3<br/>(parallel)"]
  CG --> L2{"_RUN_LOCK<br/>timed run<br/>(exclusive — one stopwatch at a time)"}
  L2 --> RES["Result: measured speedup"]
```

</details>

**Why two locks:**

- **`_ISL_LOCK`** (`backend_isl.py`) — islpy builds every object in a
  process-global ISL context that is *not* thread-safe. The legality section is
  microsecond-to-millisecond cheap, so serializing it costs almost nothing; the
  expensive compile/run stays outside it.
- **`_RUN_LOCK`** (`runner.py`) — only one *timed* binary executes at a time.
  Concurrent benchmark processes contend for cores and caches and would bias the
  very wall-clock speedup the agent optimizes for. Compilation and the LLM calls
  still overlap freely.

## Timeline: compiles overlap, runs don't

Two candidates `A` and `B` evaluated concurrently. Their legality checks and
compiles overlap; only the timed runs are exclusive, so each measurement is taken
on an otherwise-idle machine.

![parallelism timeline](images/parallelism-timeline.png)

<details><summary>mermaid source</summary>

```mermaid
sequenceDiagram
  participant T as Worker threads
  participant I as _ISL_LOCK
  participant R as _RUN_LOCK
  Note over T: candidates A, B evaluate concurrently
  T->>I: A legality
  I-->>T: legal
  T->>I: B legality (queues behind A)
  I-->>T: legal
  par compiles overlap (GIL released)
    T->>T: clang -O3 A
  and
    T->>T: clang -O3 B
  end
  T->>R: time A (exclusive)
  R-->>T: speedup A
  T->>R: time B (queues behind A's run)
  R-->>T: speedup B
```

</details>

## Correctness

`tests/test_parallel_safety.py` hammers a shared environment with hundreds of
concurrent `evaluate()` calls and asserts every schedule's legality **verdict**
matches the serial reference (speedups jitter; status must not):

```
OK: 384 concurrent evaluate() calls, 0 mismatches, verdicts identical to serial.
```

A measured illustration of the measurement lock's effect (mock GEMM, best-of-3):
without it, simultaneous timed runs depressed an honest ~50× to 8–21×; with it,
the runs still interleave (parallel search) while measurements return to ~34–54×.

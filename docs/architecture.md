# Architecture

Four views of ComPilot. Each is shown as a rendered image (works everywhere) with its live mermaid source below.

## System architecture

The agent proposes schedules; the environment proves legality (ISL) and measures real speedup (clang). The Gemini key is read from OpenBao at runtime.

![architecture](images/architecture.png)

<details><summary>mermaid source</summary>

```mermaid
flowchart LR
  subgraph Agent
    LLM["Gemini 2.5-flash<br/>(llm.py)"]
    PR["prompt.py<br/>context + nest"]
    FB["feedback.py<br/>5 categories"]
  end
  subgraph Environment["Environment (backend_isl.py)"]
    P["schedule.py<br/>parse 9 primitives"]
    SCH["scheduler.py<br/>→ ISL schedule map θ'"]
    LEG{"polyhedral.py<br/>ISL legality + parallelism"}
    CG["codegen.py<br/>→ C"]
    CC["runner.py<br/>clang -O3 +OpenMP → run"]
  end
  OB[("OpenBao<br/>secrets/google")]

  PR --> LLM
  LLM -->|schedule| P --> SCH --> LEG
  LEG -- illegal --> FB
  LEG -- legal --> CG --> CC -->|measured speedup| FB
  FB -->|next message| LLM
  OB -. api_key at runtime .-> LLM
```

</details>

## The optimization dialogue

One LLM↔compiler conversation: propose a schedule, evaluate it, feed the outcome back, keep the best legal speedup, stop on the stop-token.

![dialogue](images/dialogue.png)

<details><summary>mermaid source</summary>

```mermaid
sequenceDiagram
  participant A as agent.py
  participant L as Gemini LLM
  participant E as Environment
  A->>L: system prompt + loop nest + baseline time
  loop until stop-token or max iters
    L-->>A: reasoning + schedule block
    A->>E: evaluate(schedule)
    E->>E: ISL legality + parallelism check
    alt legal
      E->>E: codegen → clang → run
      E-->>A: success + speedup
    else illegal / cannot-parallelize / invalid / error
      E-->>A: feedback category
    end
    A->>L: feedback (+ best-so-far)
  end
  A-->>A: best speedup, best schedule
```

</details>

## Polyhedral legality (the faithful core)

A schedule is legal iff every dependence stays lexicographically forward under the new schedule θ′; a loop level is parallel iff it carries no dependence. The LLM can be wrong — this rejects illegal schedules before any code runs.

![legality](images/legality.png)

<details><summary>mermaid source</summary>

```mermaid
flowchart TD
  K["Kernel<br/>domain · reads · writes · loop order"] --> D["Dependences D<br/>RAW/WAR/WAW, ordered by original schedule"]
  S["Schedule (9 primitives)"] --> T["θ' : iteration → new time vector"]
  D --> CK{"∀ (i → i') ∈ D :<br/>θ'(i) ≺lex θ'(i') ?"}
  T --> CK
  CK -- yes --> OK["LEGAL → compile and measure"]
  CK -- no --> BAD["ILLEGAL → return violations"]
  T --> PAR{"level p carries<br/>no dependence?"}
  PAR -- yes --> Pok["parallel(p) allowed"]
  PAR -- no --> Pno["parallel(p) rejected"]
```

</details>

## Backend abstraction

The same Environment interface is served by the ISL backend (islpy + clang, done) or the real Tiramisu compiler (built, cross-validated 4/4).

![backends](images/backends.png)

<details><summary>mermaid source</summary>

```mermaid
flowchart LR
  AG["agent loop"] --> ENV["Environment interface<br/>evaluate(schedule) → Result"]
  ENV --> B1["ISL backend (done)<br/>islpy legality + clang exec"]
  ENV -.-> B2["Tiramisu backend (built)<br/>real libtiramisu legality, cross-validated 4/4"]
```

</details>

## Kernel registries

Four registries in `compilot/kernels.py`, each with its own codegen + environment.
Together they cover the full PolyBench/C 4.2.1 set (30/30) across 5 size classes.

| Registry | Shape | Codegen | Examples |
|---|---|---|---|
| `REGISTRY` | single statement, perfect nest | `codegen.py` | gemm, syrk, syr2k, floyd-warshall |
| `MULTI_REGISTRY` | a sequence of statements | `multikernel.py` | 2mm, 3mm, gemver, covariance, doitgen |
| `STENCIL_REGISTRY` | sequential time loop over spatial sweeps | `stencil.py` | jacobi-1d/2d, seidel-2d, heat-3d, fdtd-2d, adi, deriche |
| `IMPERFECT_REGISTRY` | loop-carried solvers / triangular BLAS (tree nest) | `imperfect.py` | trisolv, lu, cholesky, ludcmp, durbin, gramschmidt, trmm, symm, nussinov |

## Single-binding-statement legality model (imperfect kernels)

Imperfect kernels emit a *tree* of statements (the real C), but legality is decided
against **one** `PolyKernel` that captures the *binding* dependence. Simple, with
one sharp edge: the oracle only sees the dependences you encode. A **reduction that
accumulates into an element other than the binding statement's own write** is
invisible unless modeled explicitly.

```mermaid
flowchart TD
  ST["Imperfect kernel<br/>tree of IStmt (emitted C)"] --> CG2["codegen: per-occurrence<br/>#pragma omp parallel for"]
  ST --> PK["one PolyKernel 'poly'<br/>(the binding dependence)"]
  PK --> LEG2{"is_parallel(loop)?<br/>does poly carry a dep here?"}
  LEG2 -- "no dep in poly" --> ACC["parallel ACCEPTED"]
  ACC --> RACE{"but a real reduction<br/>accumulates off the<br/>binding's write element?"}
  RACE -- yes --> BUG["UNSOUND: race emitted<br/>(only the checksum may catch it)"]
  RACE -- no --> SAFE["sound"]
  LEG2 -- "carries dep" --> REJ["parallel REJECTED"]
```

This is the class of bug fixed in issues **#9/#10**: `symm`'s `temp2 += …` reduction
(carried on `k`) and `gramschmidt`'s `nrm`/`R[k][j]` reductions (carried on `i`)
accumulate into *different* elements than the binding write, so the oracle wrongly
accepted `parallel(k)` / `parallel(i)` until those reductions were added to `poly`
as virtual accumulator arrays (`acc[i,j]`, `R[k,j]`).

**Rule of thumb:** when adding an imperfect kernel, every loop that runs a reduction
must carry a dependence in `poly`, even if the binding statement's write does not.
The runtime checksum is a backstop, not the proof — a data race can pass a checksum
by luck, so the checksum is also **position-weighted** (`Σ (idx+1)·out[idx]`) to
catch transposed/mirrored writes that an unweighted sum would miss.

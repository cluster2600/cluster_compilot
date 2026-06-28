"""Prompts: the constant context prompt (system) and per-kernel presentation.

Mirrors the paper's Fig. 2 (context prompt: role, formats, transformation
repertoire, action space, hardware, crash handling) and Fig. 3/4 (loop nest +
request for a chain-of-thought analysis before proposing transformations).
"""
from . import codegen

STOP_TOKEN = "no_further_transformations"

SYSTEM = f"""You are a compiler optimization assistant. You optimize a single loop nest by
proposing sequences of legal loop transformations to a compiler, which checks
their legality and reports the measured speedup. You iterate using that feedback.

# Transformation repertoire (one transform per line, applied in order)
  interchange(La, Lb)            swap two loops
  reorder(La, Lb, Lc, ...)       permute the nest into this exact loop order
  tile(L, T)                     tile loop L with tile size T (creates loop L_t)
  tile2d(La, Lb, Ta, Tb)         tile two loops
  tile3d(La, Lb, Lc, Ta, Tb, Tc) tile three loops
  parallel(L)                    parallelize loop L (illegal if L carries a dependence)
  unroll(L, F)                   unroll loop L by factor F
  skew(Ltarget, Lsrc, factor)    skew Ltarget by factor*Lsrc
  reverse(L)                     reverse loop L's iteration order

Loop labels are the loop variable names shown in the nest. Tiling loop L creates
an outer tile loop named L_t that you may then reorder/parallelize.

# Action space
You may combine transformations, revoke previous ones (by re-proposing a new full
sequence), and modify sizes/factors. Each proposal is the COMPLETE schedule from
the original nest (not a delta).

# Output format
First a short "Reasoning:" explaining your strategy given the feedback so far.
Then the full schedule inside tags, e.g.:
<schedule>
reorder(i, k, j)
tile2d(i, j, 32, 32)
parallel(i_t)
</schedule>
When you believe no further useful transformation exists, output:
<schedule>{STOP_TOKEN}</schedule>

# Feedback you will receive
Legal + measured speedup; Illegal (dependence violation); Cannot-parallelize;
Invalid syntax; or Compiler/runtime error. Use it to refine the next proposal.

# Hardware
Multicore CPU with OpenMP; many threads available. Favour locality (loop order,
tiling) and thread-level parallelism on the outermost legal loop.
"""


def kernel_message(env):
    nest = codegen.render_nest(env.ek)
    t0 = env.baseline()["time"]
    return f"""Here is the loop nest to optimize:

{nest}

Baseline execution time (original, unoptimized): {t0:.4f} s.

First, analyze the nest: identify each loop's role, data dependencies, and which
loops can be parallelized or reordered. Then propose your first schedule."""


def multi_candidate_hint(n):
    """Appended to the first message when the dialogue evaluates N candidates per turn."""
    return (
        f"\n\n# Parallel evaluation (this session)\n"
        f"You may propose up to {n} DISTINCT candidate schedules in a single turn, each "
        f"in its OWN <schedule> block. They are compiled and measured in parallel and you "
        f"receive every result, so use the turn to EXPLORE different strategies at once "
        f"(e.g. different loop orders, tile sizes, or which loop to parallelize) rather "
        f"than proposing just one. Each block is still a COMPLETE schedule from the "
        f"original nest. When no further useful transformation exists, output a single "
        f"<schedule>{STOP_TOKEN}</schedule>."
    )


def kernel_message_multi(menv):
    from .multikernel import _Kernelish
    n = len(menv.mk.statements)
    parts = [f"This kernel has {n} statements, run in sequence (a later statement may read an "
             f"earlier one's output buffer). Provide exactly {n} <schedule> blocks, one per "
             f"statement, IN ORDER.\n"]
    for idx, s in enumerate(menv.mk.statements):
        nest = codegen.render_nest(_Kernelish(s.loops, s.body))
        parts.append(f"Statement {idx} (writes `{s.output}`):\n{nest}\n")
    parts.append(f"Baseline time (all statements): {menv.baseline()['time']:.4f} s.\n"
                 f"Analyze each statement, then output {n} schedule blocks in order.")
    return "\n".join(parts)

"""Imperfect-nest kernels (Track C): solvers whose statements live at DIFFERENT
loop depths, under a possibly loop-carried outer loop, with SIBLING inner loops.

PolyBench solvers (trisolv/lu/cholesky/ludcmp/durbin/gramschmidt, plus the
triangular BLAS trmm/symm) are not perfect nests. cholesky, under row i, runs a
j-loop (itself holding a k-reduction then a divide) AND a separate diagonal
k-loop AND a scalar sqrt — three siblings at differing depths. So a kernel here
is an explicit list of statements, each tagged with its OWN enclosing loop list;
codegen builds the loop tree by walking statements in program order and keeping
the common loop prefix open (open new loops, close the ones that diverge). Two
adjacent statements with identical loop tuples share the loop (fusion); give a
loop a different var name to force a sibling loop instead (durbin's z/copy).

Legality reuses the single-statement polyhedral engine on ONE binding statement
(`poly`) — the deepest/most-constrained one, which carries the dependence that
pins the schedule for this kernel family. ponytail: single-binding-statement
model. Exact when one statement constrains the schedule (the triangular solvers);
add a union-domain multi-statement model if a kernel needs a cross-statement
dependence the binding statement misses.

Execution supports parallel() (an OpenMP `parallel for` on a legal loop) and the
identity baseline; other legal transforms report "unsupported" (legality still
proven). The point of this family is the legality engine REJECTING illegal
parallelism on the sequential carried loop while ACCEPTING it on independent ones.
"""
from dataclasses import dataclass, field

from . import runner as _runner
from . import schedule as _schedule
from .backend_isl import Result
from .polyhedral import PolyKernel, dependences, is_legal, is_parallel
from .scheduler import build_theta

_EXECUTABLE = {"parallel"}        # transforms the imperfect codegen can emit


@dataclass
class IStmt:
    loops: list      # enclosing loops outermost-first: (var, lo, hi) asc, or (var, lo, hi, "rev") desc
    body: str        # raw C with explicit flat indexing


@dataclass
class ImperfectKernel:
    name: str
    sizes: dict
    arrays: dict                 # name -> tuple of dims (any rank)
    statements: list             # [IStmt] in program order
    poly: PolyKernel             # binding statement -> legality
    final: str                   # checksum array
    reset: dict = field(default_factory=dict)   # array -> "zero" | "reinit" (per rep)
    setup: str = ""              # C run each rep before timing (scalar decls, diagonal boost)


def _emit_nest(statements, parallel_vars, base):
    """Walk statements, keeping the common loop prefix open (tree codegen).

    A parallel loop gets a `#pragma omp parallel for` at each of its occurrences.
    Per-occurrence fork/join (not a hoisted region) — always correct regardless of
    sibling statements; the solver family doesn't win from parallelism anyway, so
    correctness + legality is the deliverable, not work-share overhead.
    """
    ind = lambda n: base + "  " * n
    lines, open_loops = [], []
    for st in statements:
        cp = 0
        while (cp < len(open_loops) and cp < len(st.loops)
               and tuple(open_loops[cp]) == tuple(st.loops[cp])):
            cp += 1
        while len(open_loops) > cp:                       # close diverged loops
            open_loops.pop()
            lines.append(ind(len(open_loops)) + "}")
        for lv in st.loops[cp:]:                          # open this statement's new loops
            n = len(open_loops)
            var, lo, hi = lv[0], lv[1], lv[2]
            rev = len(lv) > 3 and lv[3] == "rev"
            if var in parallel_vars:
                lines.append(ind(n) + "#pragma omp parallel for")
            if rev:
                lines.append(ind(n) + f"for(int {var}=({hi})-1;{var}>=({lo});{var}--){{")
            else:
                lines.append(ind(n) + f"for(int {var}={lo};{var}<{hi};{var}++){{")
            open_loops.append(lv)
        lines.append(ind(len(open_loops)) + st.body)
    while open_loops:
        open_loops.pop()
        lines.append(ind(len(open_loops)) + "}")
    return "\n".join(lines)


def _emit(ik, parallel_vars):
    tot = lambda a: "*".join(str(x) for x in ik.arrays[a])
    pat = "(double)((f_*13+7)%97)/97.0"
    sizes = "\n".join(f"  const int {k} = {v};" for k, v in ik.sizes.items())
    allocs = "\n".join(f"  double *{a} = malloc((size_t)({tot(a)})*sizeof(double));" for a in ik.arrays)
    inits = "\n".join(f"  for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]={pat};"
                      for a in ik.arrays if a not in ik.reset)
    resets = "\n".join(
        f"    for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]={pat if m == 'reinit' else '0.0'};"
        for a, m in ik.reset.items())
    setup = f"    {ik.setup}" if ik.setup else ""
    nest = _emit_nest(ik.statements, parallel_vars, "    ")
    # position-weighted (f_+1): an unweighted sum is permutation-invariant and would
    # miss a transposed/mirrored write that lands the right values in the wrong cells.
    checksum = f"  double acc_=0; for(long f_=0;f_<(long)({tot(ik.final)});f_++) acc_+=(double)(f_+1)*{ik.final}[f_];"
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
int main(void){{
{sizes}
{allocs}
{inits}
  double best=1e30;
  for(int rep=0;rep<3;rep++){{
{resets}
{setup}
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
{nest}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    if(dt<best)best=dt;
  }}
{checksum}
  printf("TIME %.6f\\nCHECKSUM %.6e\\n", best, acc_);
  return 0;
}}
"""


class ImperfectEnvironment:
    """Single shared schedule over an imperfect nest. Exposes .pk/.baseline()/.evaluate
    so the single-statement agent dialogue and prompt builder work unchanged."""

    def __init__(self, ik):
        self.ik = ik
        self.pk = ik.poly
        self.D = dependences(ik.poly)
        self._baseline = None

    def baseline(self):
        if self._baseline is None:
            r = _runner.compile_and_run(_emit(self.ik, set()))
            if not r["ok"]:
                raise RuntimeError(f"baseline failed: {r}")
            self._baseline = r
        return self._baseline

    def evaluate(self, schedule_text) -> Result:
        try:
            ops = _schedule.parse(schedule_text)
            theta, labels, par, _ = build_theta(self.pk, ops)
        except ValueError as e:
            return Result("invalid", detail=str(e), schedule=schedule_text)
        legal, viol = is_legal(self.D, theta)
        if not legal:
            return Result("illegal", detail=f"violates dependences: {viol}", schedule=schedule_text)
        parallel_vars = set()
        for lbl, lvl in par:
            if not is_parallel(self.D, theta, lvl):
                return Result("parallel_illegal",
                              detail=f"loop {lbl} carries a dependence; cannot parallelize",
                              schedule=schedule_text)
            parallel_vars.add(lbl)
        used = {op for op, _ in ops}
        if not used <= _EXECUTABLE:
            return Result("unsupported", detail=f"legal, but imperfect codegen lacks {used - _EXECUTABLE}",
                          schedule=schedule_text)
        base = self.baseline()
        r = _runner.compile_and_run(_emit(self.ik, parallel_vars))
        if not r["ok"]:
            return Result(r["error"], detail=r.get("detail", ""), schedule=schedule_text)
        ref = base["checksum"]
        if abs(r["checksum"] - ref) > 1e-6 * max(1.0, abs(ref)):
            return Result("incorrect", detail=f"checksum {r['checksum']:.6e} != baseline {ref:.6e}",
                          schedule=schedule_text)
        return Result("success", speedup=base["time"] / max(r["time"], 1e-9),
                      detail=f"{base['time']:.4f}s -> {r['time']:.4f}s", schedule=schedule_text)

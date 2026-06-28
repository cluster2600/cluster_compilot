"""Imperfect-nest kernels (Track C): solvers whose statements live at DIFFERENT
loop depths and whose outer loop is loop-carried (sequential).

PolyBench solvers (trisolv/lu/cholesky/ludcmp/...) are not perfect nests: a
triangular solve runs `x[i]=b[i]` (depth 1), an inner reduction (depth 2), then
`x[i]/=L[i][i]` (depth 1). The perfect-nest codegen can't express that. Here a
kernel is an explicit list of statements tagged with their loop depth, emitted by
a depth-walk that opens/closes loops as the depth changes.

Legality reuses the single-statement polyhedral engine on the DEEPEST statement's
self-dependences — it spans every loop, so it carries the binding dependences for
this kernel family (the inner reduction/update is what pins the schedule).
ponytail: deepest-statement model. Exact when the innermost statement constrains
the schedule (true for the triangular solvers here); add a union-domain
multi-statement model if a solver needs a cross-statement dependence the deepest
statement misses.

Execution supports parallel() (an OpenMP forall on a legal loop) and the identity
baseline; other legal transforms report "unsupported" (legality is still proven).
The point of this family is the legality engine REJECTING illegal parallelism on
the sequential carried loop while ACCEPTING it on the independent loops.
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
    depth: int       # number of enclosing loops (1 = inside loops[0], 2 = loops[0..1], ...)
    body: str        # raw C with explicit flat indexing


@dataclass
class ImperfectKernel:
    name: str
    sizes: dict
    arrays: dict                 # name -> tuple of dims (any rank)
    loops: list                  # canonical nest [(var, lo, hi), ...]
    statements: list             # [IStmt] in program order
    poly: PolyKernel             # deepest statement, spans all loops -> legality
    final: str                   # checksum array
    reset: dict = field(default_factory=dict)   # array -> "zero" | "reinit" (per rep)
    setup: str = ""              # C run each rep after reset, before timing (e.g. diagonal boost)


def _emit_nest(loops, statements, parallel_vars, base):
    """Depth-walk: open loops to reach each statement's depth, close on the way out.

    A parallel loop becomes ONE hoisted `#pragma omp parallel` region with an
    `omp for` work-share on that loop (implicit barrier each step), not a fresh
    fork/join per outer iteration — the latter is pure overhead for the carried
    solvers (a parallel region created N times). Outer loops run redundantly in
    every thread as pure control. ponytail: correct when the parallel loop
    encloses all real work and its outer loops are pure loop control (the solver
    family); a statement strictly between an outer loop and the parallel loop
    would need explicit single/masking.
    """
    par = [v for v, _, _ in loops if v in parallel_vars]
    worksh = par[0] if par else None          # only the outermost level is work-shared
    pad = "  " if worksh else ""
    ind = lambda n: base + pad + "  " * n
    lines, cur = [], 0
    for st in statements:
        while cur < st.depth:
            var, lo, hi = loops[cur]
            if var == worksh:
                lines.append(f"{ind(cur)}#pragma omp for")
            lines.append(f"{ind(cur)}for(int {var}={lo};{var}<{hi};{var}++){{")
            cur += 1
        while cur > st.depth:
            cur -= 1
            lines.append(f"{ind(cur)}}}")
        lines.append(f"{ind(st.depth)}{st.body}")
    while cur > 0:
        cur -= 1
        lines.append(f"{ind(cur)}}}")
    body = "\n".join(lines)
    if worksh:
        return f"{base}#pragma omp parallel\n{base}{{\n{body}\n{base}}}"
    return body


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
    nest = _emit_nest(ik.loops, ik.statements, parallel_vars, "    ")
    checksum = f"  double acc_=0; for(long f_=0;f_<(long)({tot(ik.final)});f_++) acc_+={ik.final}[f_];"
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
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
        # Codegen ceiling: the hoisted parallel region work-shares the OUTERMOST parallel
        # loop, which must enclose every statement. A shallower sibling (e.g. lu's row-scale
        # before the update loop) would run redundantly in all threads -> wrong result. The
        # dependence is legal; this codegen just can't emit it. Report it, don't emit it.
        if parallel_vars:
            order = [v for v, _, _ in self.ik.loops]
            w = min(order.index(v) for v in parallel_vars)
            if any(st.depth <= w for st in self.ik.statements):
                return Result("unsupported",
                              detail=f"legal, but single-work-share codegen can't parallelize "
                                     f"{order[w]} with a shallower sibling statement",
                              schedule=schedule_text)
        base = self.baseline()
        r = _runner.compile_and_run(_emit(self.ik, parallel_vars))
        if not r["ok"]:
            return Result(r["error"], detail=r.get("detail", ""), schedule=schedule_text)
        ref = base["checksum"]
        if abs(r["checksum"] - ref) > 1e-6 * max(1.0, abs(ref)):
            return Result("incorrect", detail=f"checksum {r['checksum']:.6e} != baseline {ref:.6e}",
                          schedule=schedule_text)
        return Result("success", speedup=base["time"] / r["time"],
                      detail=f"{base['time']:.4f}s -> {r['time']:.4f}s", schedule=schedule_text)

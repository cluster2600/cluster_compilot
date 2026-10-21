"""Multi-statement execution (Track B): kernels that are a SEQUENCE of statements.

Models PolyBench kernels like 2mm/3mm: several single-statement nests run in
program order, sharing buffers (a later statement reads an earlier one's output).
Each statement is scheduled independently; legality is the existing single-
statement check per statement (the cross-statement order is preserved by running
them in sequence). Codegen emits all nests into one timed program; the final
output's checksum guards correctness.

Loop fusion across statements (fuse / shift) runs them in a shared nest: the fused
statements then become one union.UnionProgram, checked and generated as a whole.
"""
import threading
from dataclasses import dataclass, field

from . import codegen as _cg
from . import schedule as _schedule
from . import runner as _runner
from . import union as _union
from .backend_isl import SCHEDULE_ERRORS
from .polyhedral import ISL_LOCK, PolyKernel, dependences, is_legal, is_parallel
from .scheduler import build_theta


@dataclass
class MStmt:
    """One statement (or a fused group): a loop nest + a paired polyhedral spec.

    A FUSED group is one MStmt whose `body` holds several statements' bodies in a
    shared nest, with `extra_outputs` naming the additional arrays to zero.
    """
    loops: list                 # [("i","N"), ...]
    body: str                   # one or more "...;" statements (fused)
    output: str                 # primary array written (zeroed each rep)
    poly: PolyKernel            # for legality of this statement's schedule
    reduction: set = field(default_factory=set)
    extra_outputs: list = field(default_factory=list)   # also reset (fused groups)
    reset: str = "zero"   # per-rep reset of outputs: "zero" | "reinit" (to input) | "none"


@dataclass
class MultiKernel:
    name: str
    sizes: dict
    arrays: dict                # all shared arrays: name -> (dim, dim)
    statements: list            # [MStmt, ...] in program order
    final: str                  # output array used for the checksum


def _emit_program(mk, nests, fused=None):
    """Generate one C program running every statement in order, timed. nests[i] is
    statement i's scheduled C nest (codegen.isl_nest), or None for its original nest.
    With `fused` (a union.fused_nest covering every statement), every statement's
    output reset is hoisted ahead of it — MultiEnvironment.program checks that's safe."""
    outputs = set()
    for s in mk.statements:
        outputs.add(s.output)
        outputs.update(s.extra_outputs)
    tot = lambda a: "*".join(str(x) for x in mk.arrays[a])   # total elements (any rank)
    sizes = "\n".join(f"  const int {k} = {v};" for k, v in mk.sizes.items())
    allocs = "\n".join(f"  double *{a} = malloc((size_t)({tot(a)})*sizeof(double));"
                       for a in mk.arrays)
    inits = "\n".join(f"  for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]=(double)((f_*13+7)%97)/97.0;"
                      for a in mk.arrays if a not in outputs)
    # reset="none" outputs are never reset inside the rep loop; zero them once so the
    # accumulation starts from a clean, deterministic slate instead of malloc garbage.
    none_outputs = list(dict.fromkeys(o for s in mk.statements if s.reset == "none"
                                      for o in (s.output, *s.extra_outputs)))
    inits += "\n" + "\n".join(f"  for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]=0.0;"
                              for a in none_outputs)

    body_lines = []
    for s, nest in zip(mk.statements, nests):
        if s.reset != "none":
            for outp in [s.output, *s.extra_outputs]:
                rhs = ("(double)((f_*13+7)%97)/97.0" if s.reset == "reinit" else "0.0")
                body_lines.append(f"    for (long f_=0;f_<(long)({tot(outp)});f_++) {outp}[f_]={rhs};")
        if fused is None:
            body_lines.append(nest if nest is not None else _cg.original_nest(s.loops, s.body))
    if fused is not None:
        body_lines.append(fused)

    nest = "\n".join(body_lines)
    # position-weighted via a globally-incrementing weight w_ across all outputs, so a
    # transposed/mirrored write (right values, wrong cells) changes the checksum.
    csum = ["  double acc_=0; long w_=0;"]
    for a in sorted(outputs):
        csum.append(f"  for (long f_=0;f_<(long)({tot(a)});f_++) acc_+=(double)(++w_)*{a}[f_];")
    checksum = "\n".join(csum)
    frees = "\n".join(f"  free({a});" for a in mk.arrays)
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
{_cg.PRELUDE}
int main(void){{
{sizes}
{allocs}
{inits}
  double best=1e30;
  for(int rep=0;rep<3;rep++){{
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
{nest}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    if(dt<best)best=dt;
  }}
{checksum}
  printf("TIME %.9f\\nCHECKSUM %.6e\\n", best, acc_);
{_cg.THREADS}
{frees}
  return 0;
}}
"""


@dataclass
class _Kernelish:
    loops: list
    body: str


class MultiEnvironment:
    def __init__(self, mk):
        self.mk = mk
        with ISL_LOCK:
            self.deps = [dependences(s.poly) for s in mk.statements]
        self._baseline = None
        self._baseline_lock = threading.Lock()
        self._union = {}                  # union dependences, built on first fuse()

    def baseline_program(self):
        return _emit_program(self.mk, [None] * len(self.mk.statements))

    def _fused(self, scheds):
        """The fuse()/shift() path: the whole sequence as one union. Caller holds ISL_LOCK."""
        stmts = self.mk.statements
        for t, s in enumerate(stmts):              # resets get hoisted ahead of the fused nest
            if s.reset == "none":
                continue
            outs = {s.output, *s.extra_outputs}
            for u in stmts[:t]:
                if outs & {a for a, _ in u.poly.reads + u.poly.writes}:
                    return None, {"status": "invalid", "speedup": None,
                                  "detail": f"cannot fuse: an earlier statement uses {sorted(outs)} "
                                            f"before statement {t} resets it"}
        nest, failed = _union.fused_nest([(s.poly, s.body) for s in stmts], scheds,
                                         self.mk.sizes, self.mk.arrays, self._union)
        return (None, failed) if failed else (_emit_program(self.mk, [None] * len(stmts), nest), None)

    def baseline(self):
        if self._baseline is None:
            with self._baseline_lock:
                if self._baseline is None:
                    r = _runner.compile_and_run(self.baseline_program())
                    if not r["ok"]:
                        raise RuntimeError(f"baseline failed: {r}")
                    self._baseline = r
        return self._baseline

    def program(self, scheds):
        """Per-statement legality + ISL codegen. Returns (program, None) or
        (None, failure dict). An empty schedule keeps that statement's original nest."""
        stmts = self.mk.statements
        scheds = list(scheds)[:len(stmts)] + [""] * (len(stmts) - len(scheds))
        nests = []
        with ISL_LOCK:                    # see polyhedral.ISL_LOCK; MoA evaluates pools in parallel
            try:
                if _union.wants_fusion(scheds):
                    return self._fused(scheds)
                for s, D, sched in zip(stmts, self.deps, scheds):
                    if not sched.strip():
                        nests.append(None)
                        continue
                    theta, labels, par, unroll = build_theta(s.poly, _schedule.parse(sched))
                    if not is_legal(D, theta)[0]:
                        return None, {"status": "illegal", "stmt": s.output, "speedup": None}
                    for lbl, lvl in par:
                        if not is_parallel(D, theta, lvl):
                            return None, {"status": "parallel_illegal", "stmt": s.output, "speedup": None}
                    nests.append(_cg.isl_nest(s.poly, self.mk.sizes, s.body, theta, par, unroll))
            except SCHEDULE_ERRORS as e:
                return None, {"status": "invalid", "speedup": None, "detail": str(e)}
        return _emit_program(self.mk, nests), None

    def evaluate(self, scheds):
        """scheds: list of schedule strings (one per statement)."""
        program, failed = self.program(scheds)
        if failed:
            return failed
        base = self.baseline()
        r = _runner.compile_and_run(program)
        if not r["ok"]:
            return {"status": r["error"], "speedup": None, "detail": r.get("detail", "")}
        if not _runner.checksums_match(r["checksum"], base["checksum"]):
            return {"status": "incorrect", "speedup": None}
        return {"status": "success", "speedup": base["time"] / r["time"],
                "detail": f"{base['time']:.4f}s -> {r['time']:.4f}s"}

    def confirm(self, scheds, rounds=_runner.CONFIRM_ROUNDS, interleaved=False):
        """Re-measured speedup (see runner.confirm_speedup), or None if the schedule
        set is not legal and correct."""
        program, failed = self.program(scheds)
        if failed:
            return None
        return _runner.confirm_speedup(self.baseline(), self.baseline_program(), program,
                                       rounds, interleaved)

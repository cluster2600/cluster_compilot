"""Multi-statement execution (Track B): kernels that are a SEQUENCE of statements.

Models PolyBench kernels like 2mm/3mm: several single-statement nests run in
program order, sharing buffers (a later statement reads an earlier one's output).
Each statement is scheduled independently; legality is the existing single-
statement check per statement (the cross-statement order is preserved by running
them in sequence). Codegen emits all nests into one timed program; the final
output's checksum guards correctness.

(Loop fusion across statements — running them in a shared nest — is the next
step, on top of polyhedral_multi's fusion legality.)
"""
from dataclasses import dataclass, field

from . import codegen as _cg
from . import schedule as _schedule
from . import runner as _runner
from .polyhedral import PolyKernel, dependences, is_legal, is_parallel
from .scheduler import build_theta


@dataclass
class MStmt:
    """One statement: its own loop nest (like Kernel) + a paired polyhedral spec."""
    loops: list                 # [("i","N"), ...]
    body: str
    output: str                 # array this statement writes (zeroed each rep)
    poly: PolyKernel            # for legality of this statement's schedule
    reduction: set = field(default_factory=set)


@dataclass
class MultiKernel:
    name: str
    sizes: dict
    arrays: dict                # all shared arrays: name -> (dim, dim)
    statements: list            # [MStmt, ...] in program order
    final: str                  # output array used for the checksum


def _emit_program(mk, scheds):
    """Generate one C program running every statement (scheduled) in order, timed."""
    outputs = {s.output for s in mk.statements}
    sizes = "\n".join(f"  const int {k} = {v};" for k, v in mk.sizes.items())
    allocs = "\n".join(f"  double *{a} = malloc((size_t){d[0]}*{d[1]}*sizeof(double));"
                       for a, d in mk.arrays.items())
    inits = []
    for a, (d0, d1) in mk.arrays.items():
        if a in outputs:
            continue
        inits.append(f"  for (int r_=0;r_<{d0};r_++) for (int c_=0;c_<{d1};c_++) "
                     f"{a}[r_*{d1}+c_]=(double)(((r_*7+c_*13)%97))/97.0;")
    inits = "\n".join(inits)

    body_lines = []
    for s, sched in zip(mk.statements, scheds):
        od0, od1 = mk.arrays[s.output]
        body_lines.append(f"    for (int r_=0;r_<{od0};r_++) for (int c_=0;c_<{od1};c_++) "
                          f"{s.output}[r_*{od1}+c_]=0.0;")
        # reuse the single-statement nest emitter
        fake = _Kernelish(s.loops, s.body)
        levels = _cg._build_levels(fake, _schedule.parse(sched) if sched.strip() else [])
        body_lines.append(_cg._emit_nest(fake, levels, indent="    "))

    nest = "\n".join(body_lines)
    csum = ["  double sum=0;"]
    for a in sorted(outputs):
        d0, d1 = mk.arrays[a]
        csum.append(f"  for (int r_=0;r_<{d0};r_++) for (int c_=0;c_<{d1};c_++) sum+={a}[r_*{d1}+c_];")
    checksum = "\n".join(csum)
    frees = "\n".join(f"  free({a});" for a in mk.arrays)
    return f"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MIN(a,b) ((a)<(b)?(a):(b))
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
  printf("TIME %.6f\\nCHECKSUM %.6e\\n", best, sum);
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
        self.deps = [dependences(s.poly) for s in mk.statements]
        self._baseline = None

    def baseline(self):
        if self._baseline is None:
            r = _runner.compile_and_run(_emit_program(self.mk, ["" for _ in self.mk.statements]))
            if not r["ok"]:
                raise RuntimeError(f"baseline failed: {r}")
            self._baseline = r
        return self._baseline

    def evaluate(self, scheds):
        """scheds: list of schedule strings (one per statement)."""
        # per-statement legality
        for s, D, sched in zip(self.mk.statements, self.deps, scheds):
            if not sched.strip():
                continue
            ops = _schedule.parse(sched)
            theta, labels, par, _ = build_theta(s.poly, ops)
            if not is_legal(D, theta)[0]:
                return {"status": "illegal", "stmt": s.output, "speedup": None}
            for lbl, lvl in par:
                if not is_parallel(D, theta, lvl):
                    return {"status": "parallel_illegal", "stmt": s.output, "speedup": None}
        base = self.baseline()
        r = _runner.compile_and_run(_emit_program(self.mk, scheds))
        if not r["ok"]:
            return {"status": r["error"], "speedup": None, "detail": r.get("detail", "")}
        if abs(r["checksum"] - base["checksum"]) > 1e-6 * max(1.0, abs(base["checksum"])):
            return {"status": "incorrect", "speedup": None}
        return {"status": "success", "speedup": base["time"] / r["time"],
                "detail": f"{base['time']:.4f}s -> {r['time']:.4f}s"}

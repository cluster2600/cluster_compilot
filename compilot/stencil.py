"""Stencil kernels (Track B): a sequential time loop wrapping spatial statements.

PolyBench stencils (jacobi/seidel/heat/fdtd/adi) iterate spatial sweeps over many
time steps; each step depends on the previous (the time loop is sequential, not
scheduled). The SPATIAL loops are what the agent schedules (parallelize/tile).
Arrays are reset once before the time loop, then evolved.

Structure emitted:
    for rep:
        reset arrays (once)
        for t in [0, TSTEPS):
            <spatial stmt 0 (scheduled)>
            <spatial stmt 1 (scheduled)>
    checksum(final)
"""
from dataclasses import dataclass, field

from . import codegen as _cg
from . import schedule as _schedule
from . import runner as _runner
from .multikernel import _Kernelish
from .polyhedral import PolyKernel, dependences, is_legal, is_parallel
from .scheduler import build_theta


@dataclass
class SStmt:
    loops: list                 # spatial loops, (var, lo, hi) for boundaries
    body: str
    poly: PolyKernel            # for legality of the spatial schedule


@dataclass
class StencilKernel:
    name: str
    sizes: dict                 # {"N":..., "TSTEPS":...}
    arrays: dict
    statements: list            # [SStmt] run each time step in order
    reset: dict                 # array -> "zero" | "reinit"  (once, before time loop)
    final: str
    tsteps: str = "TSTEPS"


def _emit(sk, scheds):
    # rank-agnostic: arrays of any rank flatten via the product of their dims, so
    # the same emitter handles 1-D (jacobi-1d), 2-D (jacobi/seidel) and 3-D
    # (heat-3d). Spatial bodies already carry explicit flat indexing.
    tot = lambda a: "*".join(str(x) for x in sk.arrays[a])
    pat = "(double)((f_*13+7)%97)/97.0"
    sizes = "\n".join(f"  const int {k} = {v};" for k, v in sk.sizes.items())
    allocs = "\n".join(f"  double *{a} = malloc((size_t)({tot(a)})*sizeof(double));"
                       for a in sk.arrays)
    reset_keys = set(sk.reset)
    inits = "\n".join(f"  for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]={pat};"
                      for a in sk.arrays if a not in reset_keys)
    resets = "\n".join(
        f"    for (long f_=0;f_<(long)({tot(a)});f_++) {a}[f_]={pat if mode == 'reinit' else '0.0'};"
        for a, mode in sk.reset.items())
    spatial = []
    for s, sched in zip(sk.statements, scheds):
        fake = _Kernelish(s.loops, s.body)
        levels = _cg._build_levels(fake, _schedule.parse(sched) if sched.strip() else [])
        spatial.append(_cg._emit_nest(fake, levels, indent="      "))
    spatial = "\n".join(spatial)
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
{resets}
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int t=0;t<{sk.tsteps};t++){{
{spatial}
    }}
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double dt=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
    if(dt<best)best=dt;
  }}
  double sum=0; for(long f_=0;f_<(long)({tot(sk.final)});f_++) sum+=(double)(f_+1)*{sk.final}[f_];  // position-weighted: catches transposed/mirrored writes
  printf("TIME %.6f\\nCHECKSUM %.6e\\n", best, sum);
  return 0;
}}
"""


class StencilEnvironment:
    def __init__(self, sk):
        self.sk = sk
        self.mk = sk            # alias so the multi-statement agent dialogue works unchanged
        self.deps = [dependences(s.poly) for s in sk.statements]
        self._baseline = None

    def baseline(self):
        if self._baseline is None:
            r = _runner.compile_and_run(_emit(self.sk, ["" for _ in self.sk.statements]))
            if not r["ok"]:
                raise RuntimeError(f"baseline failed: {r}")
            self._baseline = r
        return self._baseline

    def evaluate(self, scheds):
        try:
            for s, D, sched in zip(self.sk.statements, self.deps, scheds):
                if not sched.strip():
                    continue
                theta, labels, par, _ = build_theta(s.poly, _schedule.parse(sched))
                if not is_legal(D, theta)[0]:
                    return {"status": "illegal", "speedup": None}
                for lbl, lvl in par:
                    if not is_parallel(D, theta, lvl):
                        return {"status": "parallel_illegal", "speedup": None}
            program = _emit(self.sk, scheds)
        except (ValueError, KeyError) as e:
            return {"status": "invalid", "speedup": None, "detail": str(e)}
        base = self.baseline()
        r = _runner.compile_and_run(program)
        if not r["ok"]:
            return {"status": r["error"], "speedup": None, "detail": r.get("detail", "")}
        if abs(r["checksum"] - base["checksum"]) > 1e-6 * max(1.0, abs(base["checksum"])):
            return {"status": "incorrect", "speedup": None}
        return {"status": "success", "speedup": base["time"] / max(r["time"], 1e-9),
                "detail": f"{base['time']:.4f}s -> {r['time']:.4f}s"}

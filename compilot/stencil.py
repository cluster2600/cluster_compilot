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
    sizes = "\n".join(f"  const int {k} = {v};" for k, v in sk.sizes.items())
    allocs = "\n".join(f"  double *{a} = malloc((size_t){d[0]}*{d[1]}*sizeof(double));"
                       for a, d in sk.arrays.items())
    reset_keys = set(sk.reset)
    inits = "\n".join(
        f"  for (int r_=0;r_<{d0};r_++) for (int c_=0;c_<{d1};c_++) "
        f"{a}[r_*{d1}+c_]=(double)(((r_*7+c_*13)%97))/97.0;"
        for a, (d0, d1) in sk.arrays.items() if a not in reset_keys)
    resets = []
    for a, mode in sk.reset.items():
        d0, d1 = sk.arrays[a]
        rhs = "(double)(((r_*7+c_*13)%97))/97.0" if mode == "reinit" else "0.0"
        resets.append(f"    for (int r_=0;r_<{d0};r_++) for (int c_=0;c_<{d1};c_++) {a}[r_*{d1}+c_]={rhs};")
    resets = "\n".join(resets)
    spatial = []
    for s, sched in zip(sk.statements, scheds):
        fake = _Kernelish(s.loops, s.body)
        levels = _cg._build_levels(fake, _schedule.parse(sched) if sched.strip() else [])
        spatial.append(_cg._emit_nest(fake, levels, indent="      "))
    spatial = "\n".join(spatial)
    fd0, fd1 = sk.arrays[sk.final]
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
  double sum=0; for(int x=0;x<{fd0};x++) for(int y=0;y<{fd1};y++) sum+={sk.final}[x*{fd1}+y];
  printf("TIME %.6f\\nCHECKSUM %.6e\\n", best, sum);
  return 0;
}}
"""


class StencilEnvironment:
    def __init__(self, sk):
        self.sk = sk
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
        return {"status": "success", "speedup": base["time"] / r["time"],
                "detail": f"{base['time']:.4f}s -> {r['time']:.4f}s"}

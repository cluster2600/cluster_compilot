"""Turn (kernel, schedule) into a self-contained, timed C program.

The transformed nest is built by rewriting an ordered list of loop *levels*.
Legality is NOT checked here — the generated program prints a checksum, and the
runner compares it against the untransformed baseline. That comparison is the
legality gate (the paper delegates legality to a polyhedral compiler; with a
plain C backend we enforce it numerically instead).
"""
from dataclasses import dataclass, field
from . import schedule as _schedule

REPS = 3  # timing repetitions; report the best


@dataclass
class Level:
    var: str
    lo: str
    hi: str
    step: str = "1"
    pragmas: list = field(default_factory=list)
    rev: bool = False


def _build_levels(kernel, ops):
    bound = {v: b for v, b in kernel.loops}          # var -> bound name
    levels = [Level(v, "0", b) for v, b in kernel.loops]

    def find(var):
        for i, lv in enumerate(levels):
            if lv.var == var:
                return i
        raise ValueError(f"loop {var!r} not in nest (current: {[l.var for l in levels]})")

    def do_tile(var, t):
        i = find(var)
        inner = levels[i]
        outer = Level(f"{var}_t", inner.lo, inner.hi, str(t))
        inner.lo = f"{var}_t"
        inner.hi = f"MIN({var}_t + {t}, {bound.get(var, inner.hi)})"
        inner.step = "1"
        levels[i:i + 1] = [outer, inner]

    for op, args in ops:
        if op == "reorder":
            if sorted(args) != sorted(l.var for l in levels):
                raise ValueError(f"reorder{tuple(args)} is not a permutation of {[l.var for l in levels]}")
            levels.sort(key=lambda lv: args.index(lv.var))
        elif op == "interchange":
            ia, ib = find(args[0]), find(args[1])
            levels[ia], levels[ib] = levels[ib], levels[ia]
        elif op == "tile":
            do_tile(args[0], args[1])
        elif op == "tile2d":
            do_tile(args[0], args[2]); do_tile(args[1], args[3])
        elif op == "tile3d":
            do_tile(args[0], args[3]); do_tile(args[1], args[4]); do_tile(args[2], args[5])
        elif op == "parallel":
            levels[find(args[0])].pragmas.append("#pragma omp parallel for")
        elif op == "unroll":
            levels[find(args[0])].pragmas.append(f"#pragma clang loop unroll_count({args[1]})")
        elif op == "reverse":
            levels[find(args[0])].rev = True
        else:
            raise ValueError(f"codegen cannot emit {op!r} yet")
    return levels


def _emit_nest(kernel, levels, indent="    "):
    out, pad = [], indent
    for lv in levels:
        for p in lv.pragmas:
            out.append(pad + p)
        if lv.rev:
            out.append(pad + f"for (int {lv.var} = ({lv.hi}) - 1; {lv.var} >= ({lv.lo}); {lv.var} -= {lv.step}) {{")
        else:
            out.append(pad + f"for (int {lv.var} = {lv.lo}; {lv.var} < {lv.hi}; {lv.var} += {lv.step}) {{")
        pad += "    "
    out.append(pad + kernel.body)
    for _ in levels:
        pad = pad[:-4]
        out.append(pad + "}")
    return "\n".join(out)


def render_nest(kernel):
    """Human-readable view of the original loop nest (for presenting to the LLM)."""
    levels = [Level(v, "0", b) for v, b in kernel.loops]
    return _emit_nest(kernel, levels, indent="")


def generate_c(kernel, schedule_text=""):
    ops = _schedule.parse(schedule_text) if schedule_text.strip() else []
    levels = _build_levels(kernel, ops)
    nest = _emit_nest(kernel, levels)

    sizes = "\n".join(f"  const int {k} = {v};" for k, v in kernel.sizes.items())
    allocs = "\n".join(
        f"  double *{a} = malloc((size_t){d[0]} * {d[1]} * sizeof(double));"
        for a, d in kernel.arrays.items()
    )
    # deterministic init for every input array (the output gets zeroed per rep)
    inits = []
    for a, (d0, d1) in kernel.arrays.items():
        if a == kernel.output:
            continue
        inits.append(
            f"  for (int x = 0; x < {d0}; x++) for (int y = 0; y < {d1}; y++) "
            f"{a}[x*{d1} + y] = (double)(((x*7 + y*13) % 97)) / 97.0;"
        )
    inits = "\n".join(inits)
    od0, od1 = kernel.arrays[kernel.output]
    zero = (f"    for (int x = 0; x < {od0}; x++) for (int y = 0; y < {od1}; y++) "
            f"{kernel.output}[x*{od1} + y] = 0.0;")
    checksum = (f"  double sum = 0.0;\n"
                f"  for (int x = 0; x < {od0}; x++) for (int y = 0; y < {od1}; y++) "
                f"sum += {kernel.output}[x*{od1} + y];")
    frees = "\n".join(f"  free({a});" for a in kernel.arrays)

    return f"""#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#define MIN(a,b) ((a) < (b) ? (a) : (b))

int main(void) {{
{sizes}
{allocs}
{inits}
  double best = 1e30;
  for (int rep = 0; rep < {REPS}; rep++) {{
{zero}
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
{nest}
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    if (dt < best) best = dt;
  }}
{checksum}
  printf("TIME %.6f\\nCHECKSUM %.6e\\n", best, sum);
{frees}
  return 0;
}}
"""

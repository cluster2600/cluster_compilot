"""Codegen comes from the proven schedule (ISL AST generation), so what runs is
exactly what the legality engine checked.

Guards the fixes that came with switching codegen to ISL:
  - a bad loop label in any op is an `invalid` schedule, never an exception, and a
    dialogue survives an LLM typo (it used to crash the whole run)
  - tile2d/tile3d are real tiles: tile loops outermost, in Tiramisu's order
    (they used to strip-mine each loop in place, i.e. no blocking)
  - transforms the old hand-written emitter got wrong now execute correctly:
    skew + interchange (wavefronts; was a compile error), reverse of a tile loop
    (silently skipped iterations), reverse-then-tile
  - every statement's polyhedral domain equals its C loop nest, and every access
    to an array the statement writes is declared in its spec — seidel2d used to
    omit its 4 diagonal neighbours, which let the racy j+i wavefront through

    python3 -m tests.test_codegen_isl
"""
import re

import islpy as isl

from compilot.agent import run_dialogue
from compilot.backend_isl import environment
from compilot.kernels import REGISTRY, MULTI_REGISTRY, STENCIL_REGISTRY, sized_kernel
from compilot.polyhedral import PolyKernel
from compilot.schedule import parse
from compilot.scheduler import build_theta
from compilot.stencil import StencilEnvironment


def test_bad_labels_are_invalid_not_crash():
    env = environment("gemm", "SMALL")
    for bad in ("unroll(zz, 4)", "reorder(i, j, k, zz)", "reorder(i, j)", "parallel(zz)",
                "skew(i, i, 1)", "skew(i, j, N)", "tile(i, 8)\ntile(i, 8)", "tile2d(i, i, 8, 8)",
                "tile(i, 8)\nskew(i_t, j, 1)"):
        r = env.evaluate(bad)
        assert r.status == "invalid", f"{bad!r}: {r.status} ({r.detail[:80]})"
    print("OK bad loop labels / malformed ops -> invalid, no exception")


class _TypoLLM:
    """Proposes a schedule with a misspelled loop once, then stops."""
    in_tokens = out_tokens = 0

    def __init__(self):
        self.n = 0

    def chat(self, system, messages):
        self.n += 1
        if self.n == 1:
            return "<schedule>\nreorder(i, k, j)\nunroll(jj, 4)\n</schedule>"
        return "<schedule>no_further_transformations</schedule>"


def test_dialogue_survives_llm_typo():
    sp, sched, trace = run_dialogue(environment("gemm", "SMALL"), _TypoLLM(), max_iters=3, verbose=False)
    assert trace and trace[0][1] == "invalid", trace
    print("OK dialogue survives an LLM typo (reported invalid, run continues)")


def _labels(name, sched):
    return build_theta(REGISTRY[name][1], parse(sched))[1]


def test_tile2d_is_a_real_tile():
    assert _labels("gemm", "reorder(i, k, j)\ntile2d(i, j, 32, 32)") == ["i_t", "j_t", "i", "k", "j"]
    assert _labels("gemm", "tile3d(i, j, k, 8, 8, 8)") == ["i_t", "j_t", "k_t", "i", "j", "k"]
    assert _labels("gemm", "tile2d(j, i, 8, 8)") == ["j_t", "i_t", "i", "j", "k"]   # argument order
    assert _labels("gemm", "tile(j, 8)") == ["i", "j_t", "j", "k"]                   # single = strip-mine
    print("OK tile2d/tile3d put the tile loops outermost (Tiramisu order)")


def test_transforms_execute_as_proven():
    gemm = environment("gemm", "SMALL")
    for sched in ("skew(j, i, 1)\ninterchange(i, j)\nparallel(j)",          # was: compile_error
                  "tile(i, 16)\nreverse(i_t)",                               # was: incorrect
                  "reverse(i)\ntile(i, 16)",
                  "reorder(i, k, j)\ntile2d(i, j, 16, 16)\nparallel(i_t)\nunroll(j, 4)",
                  "tile3d(i, j, k, 16, 16, 16)\nreorder(i_t, j_t, k_t, i, k, j)\nparallel(i_t)"):
        r = gemm.evaluate(sched)
        assert r.status == "success", f"gemm {sched!r}: {r.status} ({r.detail[:160]})"
    # wavefront: t carries the recurrence, so after skewing t by s the inner s loop is parallel
    mc = environment("montecarlo", "SMALL").evaluate("skew(t, s, 1)\ninterchange(s, t)\nparallel(s)")
    assert mc.status == "success", (mc.status, mc.detail[:160])
    # seidel-2d's 9-point dependences need the j+2i wavefront; j+i is a race
    se = StencilEnvironment(sized_kernel("seidel2d", "SMALL"))
    assert se.evaluate(["skew(j, i, 1)\ninterchange(i, j)\nparallel(i)"])["status"] == "parallel_illegal"
    r = se.evaluate(["skew(j, i, 2)\ninterchange(i, j)\nparallel(i)"])
    assert r["status"] == "success", r
    print("OK skew wavefronts / reversed tiles / real tiles execute and match the baseline checksum")


def test_old_seidel_spec_would_have_allowed_a_race():
    """Documents why the access check below exists."""
    from compilot.polyhedral import dependences, is_parallel
    four_point = PolyKernel("A", ["i", "j"], "1<=i<N-1 and 1<=j<N-1", [("A", "i,j")],
                            [("A", "i-1,j"), ("A", "i,j-1"), ("A", "i+1,j"), ("A", "i,j+1"), ("A", "i,j")],
                            ["N"])
    th, _, par, _ = build_theta(four_point, parse("skew(j, i, 1)\ninterchange(i, j)\nparallel(i)"))
    assert is_parallel(dependences(four_point), th, par[0][1])     # wrongly "parallel"
    print("OK (documented) a 4-point seidel spec accepts the racy j+i wavefront")


def _statements():
    for name in REGISTRY:
        ek, pk = sized_kernel(name)
        yield name, ek.loops, ek.body, pk, ek.arrays, ek.sizes
    for reg in (MULTI_REGISTRY, STENCIL_REGISTRY):
        for name in reg:
            mk = sized_kernel(name)
            for i, s in enumerate(mk.statements):
                yield f"{name}[{i}]", s.loops, s.body, s.poly, mk.arrays, mk.sizes


def test_poly_domains_match_loops():
    """Codegen scans the polyhedral domain; the baseline runs the C loops. They must agree."""
    n = 0
    for label, loops, _, pk, _, sizes in _statements():
        assert [l[0] for l in loops] == list(pk.order), (label, loops, pk.order)
        params = sorted(set(pk.params) | set(sizes))
        cons = " and ".join(f"{l[1]} <= {l[0]} < {l[2]}" if len(l) == 3 else f"0 <= {l[0]} < {l[1]}"
                            for l in loops)
        ctx = isl.Set(f"[{','.join(params)}] -> {{ : "
                      + " and ".join(f"{k} = {v}" for k, v in sizes.items()) + " }")
        c_loops = isl.Set(f"[{','.join(params)}] -> {{ S[{','.join(pk.order)}] : {cons} }}")
        poly = isl.Set(f"[{','.join(params)}] -> {{ S[{','.join(pk.order)}] : {pk.domain} }}")
        assert c_loops.intersect_params(ctx).is_equal(poly.intersect_params(ctx)), label
        n += 1
    print(f"OK {n} statements: polyhedral domain == C loop nest")


_ACC = re.compile(r"\b([A-Za-z_]\w*)\[([^\[\]]+)\]")
_WRITE = re.compile(r"\b([A-Za-z_]\w*)\[([^\[\]]+)\]\s*[+\-*/]?=(?!=)")
_LOCAL = re.compile(r"\bint\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")


def _aff(expr, iters, sizes):
    for k in sorted(sizes, key=len, reverse=True):
        expr = re.sub(rf"\b{k}\b", str(sizes[k]), expr)
    return isl.PwAff(f"{{ [{','.join(iters)}] -> [({expr})] }}")


def _flat(idx, dims, iters, sizes):
    parts = [p.strip() for p in idx.split(",")]
    e = f"({parts[0]})"
    for p, d in zip(parts[1:], list(dims)[-len(parts):][1:]):
        e = f"({e})*{sizes.get(d, d)} + ({p})"
    return _aff(e, iters, sizes)


def test_written_array_accesses_are_declared():
    """Any access in the C body to an array that statement writes (and every write)
    must appear in its polyhedral spec — otherwise dependences go missing and an
    illegal schedule can pass. Read-only arrays can't create intra-statement deps."""
    n = 0
    for label, _, body, pk, arrays, sizes in _statements():
        for name, expr in _LOCAL.findall(body):         # inline `int j=N-2-jj;` (backward sweeps)
            body = re.sub(rf"\bint\s+{name}\s*=\s*{re.escape(expr)};", "", body)
            body = re.sub(rf"\b{name}\b", f"({expr})", body)
        it = list(pk.order)
        declared = [(a, _flat(i, arrays[a], it, sizes)) for a, i in pk.reads + pk.writes]
        dwrites = [(a, _flat(i, arrays[a], it, sizes)) for a, i in pk.writes]
        written = {a for a, _ in _WRITE.findall(body)} | {a for a, _ in pk.writes}
        for a, e in _ACC.findall(body):
            if a in written:
                f = _aff(e, it, sizes)
                assert any(a == b and f.is_equal(g) for b, g in declared), f"{label}: undeclared {a}[{e}]"
        for a, e in _WRITE.findall(body):
            f = _aff(e, it, sizes)
            assert any(a == b and f.is_equal(g) for b, g in dwrites), f"{label}: undeclared write {a}[{e}]"
        n += 1
    print(f"OK {n} statements: every access to a written array is in the polyhedral spec")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
    print(f"\ntest_codegen_isl: all {len(_TESTS)} checks passed")

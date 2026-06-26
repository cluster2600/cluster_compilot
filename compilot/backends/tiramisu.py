"""Exact Tiramisu backend (Track A): drive the real polyhedral compiler.

For a (kernel, schedule), emit a C++ Tiramisu program that builds the kernel,
applies the schedule via Tiramisu's API, and runs Tiramisu's own
`check_legality_of_function()` (polyhedral dependence analysis) + parallelization
legality. Compile it against the libtiramisu we built, run it, parse the verdict.

This is the exact mechanism the paper uses. We cross-validate its verdict against
our ISL engine (backend_isl) — they should agree.

Currently: GEMM, legality for interchange/tile2d/parallel/unroll/reverse. Codegen+
execution via Tiramisu's Halide path is the next sub-step (we already measure
speedup with clang in backend_isl).
"""
import os
import re
import subprocess
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "tiramisu")
ROOT = os.path.abspath(ROOT)
HALIDE = os.path.join(ROOT, "3rdParty", "Halide", "install")
ISL = os.path.join(ROOT, "3rdParty", "isl", "build")
BUILD = os.path.join(ROOT, "build")


def _schedule_cpp(ops):
    """Translate schedule ops into Tiramisu API calls.

    GEMM is two computations (C_init over {i,j}, C over {i,j,k}). An i/j-only
    transform must be applied to BOTH so they stay aligned; a k transform applies
    only to C. Otherwise the init-before-accumulate dependence is violated.
    """
    lines, newvars = [], []

    def emit(call, loop_vars):
        targets = ["C_init", "C"] if set(loop_vars) <= {"i", "j"} else ["C"]
        for t in targets:
            lines.append(f"    {t}.{call};")

    for op, a in ops:
        if op == "interchange":
            emit(f"interchange({a[0]}, {a[1]})", [a[0], a[1]])
        elif op == "reorder":
            cur = ["i", "j", "k"]
            for pos, want in enumerate(list(a)):
                src = cur.index(want)
                if src != pos:
                    emit(f"interchange({cur[pos]}, {cur[src]})", [cur[pos], cur[src]])
                    cur[pos], cur[src] = cur[src], cur[pos]
        elif op == "tile2d":
            o = [f"{a[0]}1", f"{a[1]}1", f"{a[0]}2", f"{a[1]}2"]
            newvars += o
            emit(f"tile({a[0]}, {a[1]}, {a[2]}, {a[3]}, {o[0]}, {o[1]}, {o[2]}, {o[3]})", [a[0], a[1]])
        elif op == "parallel":
            emit(f"parallelize({a[0]})", [a[0]])
        elif op == "unroll":
            emit(f"unroll({a[0]}, {a[1]})", [a[0]])
        elif op == "reverse":
            newvars.append(f"{a[0]}_r")
            emit(f"loop_reversal({a[0]}, {a[0]}_r)", [a[0]])
        else:
            return None, None  # unsupported in this bridge yet
    return lines, newvars


def gemm_program(ops, obj_path=None, fname="gemm"):
    sched_lines, newvars = _schedule_cpp(ops)
    if sched_lines is None:
        return None
    cg = (f'    if (legal) {{ tiramisu::codegen({{&b_A, &b_B, &b_C}}, "{obj_path}"); '
          f'printf("CODEGEN_OK\\n"); }}\n') if obj_path else ""
    decl_new = "".join(f' var {v}("{v}");\n' for v in newvars)
    sched = "\n".join(sched_lines)
    return f"""#include <tiramisu/tiramisu.h>
using namespace tiramisu;
int main() {{
    tiramisu::init("{fname}");
    function *fct = global::get_implicit_function();
    int N = 64, M = 64, K = 64;
    var i("i", 0, N), j("j", 0, M), k("k", 0, K);
{decl_new}
    input A("A", {{i, k}}, p_float64);
    input B("B", {{k, j}}, p_float64);
    computation C_init("C_init", {{i, j}}, expr((double)0));
    computation C("C", {{i, j, k}}, p_float64);
    C.set_expression(C(i, j, k - 1) + A(i, k) * B(k, j));
    C_init.then(C, j);

    buffer b_A("b_A", {{N, K}}, p_float64, a_input);
    buffer b_B("b_B", {{K, M}}, p_float64, a_input);
    buffer b_C("b_C", {{N, M}}, p_float64, a_output);
    A.store_in(&b_A);
    B.store_in(&b_B);
    C_init.store_in(&b_C);
    C.store_in(&b_C, {{i, j}});

    perform_full_dependency_analysis();
{sched}
    prepare_schedules_for_legality_checks();
    bool legal = check_legality_of_function();
    printf("LEGAL %d\\n", legal ? 1 : 0);
{cg}    return 0;
}}
"""


def codegen(schedule_ops):
    """Have Tiramisu generate a Halide object for a (legal) scheduled GEMM.

    Returns (object_size_bytes, info) or (None, error). Proves the real compiler
    lowers our scheduled kernel through its Halide codegen path.
    """
    d = tempfile.mkdtemp(prefix="tira_cg_")
    obj = os.path.join(d, "gemm.o")
    src = gemm_program(schedule_ops, obj_path=obj)
    if src is None:
        return None, "unsupported in tiramisu bridge"
    try:
        cpp, binp = os.path.join(d, "g.cpp"), os.path.join(d, "g")
        with open(cpp, "w") as f:
            f.write(src)
        cc = ["clang++", "-std=c++17", cpp, "-o", binp,
              f"-I{ROOT}/include", f"-I{HALIDE}/include", f"-I{ISL}/include",
              f"-L{BUILD}", "-ltiramisu", f"-L{HALIDE}/lib", "-lHalide", f"-L{ISL}/lib", "-lisl",
              f"-Wl,-rpath,{BUILD}", f"-Wl,-rpath,{HALIDE}/lib", f"-Wl,-rpath,{ISL}/lib"]
        cp = subprocess.run(cc, capture_output=True, text=True, timeout=180)
        if cp.returncode != 0:
            return None, "compile_error: " + cp.stderr[-500:]
        rp = subprocess.run([binp], capture_output=True, text=True, timeout=180, cwd=d)
        if "CODEGEN_OK" in rp.stdout and os.path.exists(obj):
            return os.path.getsize(obj), "ok"
        return None, "codegen_failed: " + (rp.stdout + rp.stderr)[-400:]
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


_LIBFLAGS = None


def _flags():
    global _LIBFLAGS
    if _LIBFLAGS is None:
        _LIBFLAGS = [f"-I{ROOT}/include", f"-I{HALIDE}/include", f"-I{ISL}/include",
                     f"-L{BUILD}", "-ltiramisu", f"-L{HALIDE}/lib", "-lHalide", f"-L{ISL}/lib", "-lisl",
                     f"-Wl,-rpath,{BUILD}", f"-Wl,-rpath,{HALIDE}/lib", f"-Wl,-rpath,{ISL}/lib"]
    return _LIBFLAGS


def _gen_object(d, ops, fname):
    """Run a Tiramisu generator to emit <fname>.o; return obj path or None."""
    obj = os.path.join(d, f"{fname}.o")
    src = gemm_program(ops, obj_path=obj, fname=fname)
    if src is None:
        return None
    cpp, binp = os.path.join(d, f"{fname}_gen.cpp"), os.path.join(d, f"{fname}_gen")
    with open(cpp, "w") as f:
        f.write(src)
    if subprocess.run(["clang++", "-std=c++17", cpp, "-o", binp] + _flags(),
                      capture_output=True, text=True, timeout=180).returncode != 0:
        return None
    subprocess.run([binp], capture_output=True, text=True, timeout=180, cwd=d)
    return obj if os.path.exists(obj) else None


_WRAPPER = r"""#include "Halide.h"
#include <tiramisu/utils.h>
#include <chrono>
#include <cstdio>
extern "C" int gemm_base(halide_buffer_t*, halide_buffer_t*, halide_buffer_t*);
extern "C" int gemm_sched(halide_buffer_t*, halide_buffer_t*, halide_buffer_t*);
int main() {
    int N = 64;
    Halide::Buffer<double> A(N, N), B(N, N), C(N, N);
    init_buffer(A, (double)1); init_buffer(B, (double)1); init_buffer(C, (double)0);
    double tb = 1e30, ts = 1e30;
    for (int r = 0; r < 7; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        gemm_base(A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count(); if (dt < tb) tb = dt;
    }
    for (int r = 0; r < 7; r++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        gemm_sched(A.raw_buffer(), B.raw_buffer(), C.raw_buffer());
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count(); if (dt < ts) ts = dt;
    }
    printf("TIME_BASE %.8f\nTIME_SCHED %.8f\n", tb, ts);
    return 0;
}
"""


def speedup(schedule_ops):
    """Speedup measured by Tiramisu's OWN Halide-generated code (baseline vs scheduled .o)."""
    import shutil
    d = tempfile.mkdtemp(prefix="tira_sp_")
    try:
        base = _gen_object(d, [], "gemm_base")
        sched = _gen_object(d, schedule_ops, "gemm_sched")
        if not base or not sched:
            return None, "codegen_failed (illegal or unsupported schedule)"
        cpp, binp = os.path.join(d, "wrap.cpp"), os.path.join(d, "wrap")
        with open(cpp, "w") as f:
            f.write(_WRAPPER)
        cp = subprocess.run(["clang++", "-std=c++17", "-O2", cpp, base, sched, "-o", binp] + _flags(),
                            capture_output=True, text=True, timeout=180)
        if cp.returncode != 0:
            return None, "wrapper_compile_error: " + cp.stderr[-500:]
        rp = subprocess.run([binp], capture_output=True, text=True, timeout=180, cwd=d)
        m = re.search(r"TIME_BASE\s+([0-9.eE+-]+)\nTIME_SCHED\s+([0-9.eE+-]+)", rp.stdout)
        if not m:
            return None, "run_failed: " + (rp.stdout + rp.stderr)[-500:]
        tb, ts = float(m.group(1)), float(m.group(2))
        return tb / ts if ts > 0 else None, f"{tb*1e6:.1f}us -> {ts*1e6:.1f}us"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def legality(schedule_ops):
    """Compile+run a Tiramisu program; return True/False/None(unsupported/error)."""
    src = gemm_program(schedule_ops)
    if src is None:
        return None, "unsupported in tiramisu bridge"
    d = tempfile.mkdtemp(prefix="tira_")
    try:
        cpp = os.path.join(d, "g.cpp")
        binp = os.path.join(d, "g")
        with open(cpp, "w") as f:
            f.write(src)
        cc = ["clang++", "-std=c++17", cpp, "-o", binp,
              f"-I{ROOT}/include", f"-I{HALIDE}/include", f"-I{ISL}/include",
              f"-L{BUILD}", "-ltiramisu", f"-L{HALIDE}/lib", "-lHalide",
              f"-L{ISL}/lib", "-lisl",
              f"-Wl,-rpath,{BUILD}", f"-Wl,-rpath,{HALIDE}/lib", f"-Wl,-rpath,{ISL}/lib"]
        cp = subprocess.run(cc, capture_output=True, text=True, timeout=180)
        if cp.returncode != 0:
            return None, "compile_error: " + cp.stderr[-600:]
        rp = subprocess.run([binp], capture_output=True, text=True, timeout=120)
        m = re.search(r"LEGAL\s+([01])", rp.stdout)
        if not m:
            return None, "no_verdict: " + (rp.stdout + rp.stderr)[-600:]
        return m.group(1) == "1", "ok"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)

"""Compile a generated C program with clang + OpenMP, run it, parse TIME/CHECKSUM.

This is the *measurement* half of the environment: real wall-clock speedup, which
the paper gets from executing Tiramisu-generated code. We execute clang-generated
code instead — the measured number is equally real.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading

_OMP = None

# Only one timed binary executes at a time, even when many candidates / best-of-k
# runs are evaluated in parallel. Concurrent benchmark processes contend for cores
# and caches and would bias the very wall-clock speedup the agent optimizes for.
# Compilation and the LLM calls still overlap freely — only the measurement is
# serialized — so the search is parallel while the numbers stay trustworthy.
_RUN_LOCK = threading.Lock()


def _libomp_prefix():
    global _OMP
    if _OMP is None:
        try:
            _OMP = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        except Exception:
            _OMP = ""
    return _OMP


def compile_and_run(c_code, threads=None, timeout=120):
    """Return dict(ok, time, checksum, error). One compile + one run."""
    omp = _libomp_prefix()
    d = tempfile.mkdtemp(prefix="compilot_")
    try:
        src = os.path.join(d, "k.c")
        binp = os.path.join(d, "k")
        with open(src, "w") as f:
            f.write(c_code)
        if omp:   # macOS Homebrew libomp: Apple clang needs -Xclang + explicit libomp
            cc = ["clang", "-O3", "-std=c11", "-Xclang", "-fopenmp", f"-I{omp}/include",
                  src, "-o", binp, "-lm", f"-L{omp}/lib", "-lomp"]
        else:     # Linux/LLVM clang: the -fopenmp driver flag enables AND links libomp.
            # glibc hides clock_gettime under -std=c11 without the POSIX feature macro.
            cc = ["clang", "-O3", "-std=c11", "-D_POSIX_C_SOURCE=199309L",
                  "-fopenmp", src, "-o", binp, "-lm"]
        try:
            cp = subprocess.run(cc, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "detail": "compile timed out"}
        if cp.returncode != 0:
            return {"ok": False, "error": "compile_error", "detail": cp.stderr[-800:]}
        env = dict(os.environ)
        if omp:
            env["DYLD_LIBRARY_PATH"] = f"{omp}/lib:" + env.get("DYLD_LIBRARY_PATH", "")
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        try:
            with _RUN_LOCK:                   # serialize timed runs; see _RUN_LOCK
                rp = subprocess.run([binp], capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout", "detail": "run timed out"}
        if rp.returncode != 0:
            return {"ok": False, "error": "runtime_error", "detail": rp.stderr[-800:]}
        t = re.search(r"TIME\s+([0-9.eE+-]+)", rp.stdout)
        c = re.search(r"CHECKSUM\s+([0-9.eE+-]+)", rp.stdout)
        if not (t and c):
            return {"ok": False, "error": "no_output", "detail": rp.stdout[-400:]}
        secs = float(t.group(1))
        # A non-positive time means the work rounded to zero at our print precision.
        # Don't divide by it (that manufactures a giant fake speedup) -- flag it so the
        # caller reruns at a larger size. The chokepoint for all four environments.
        if secs <= 0:
            return {"ok": False, "error": "measurement",
                    "detail": f"measured time {secs}s too small to trust; use a larger size class"}
        return {"ok": True, "time": secs, "checksum": float(c.group(1))}
    finally:
        shutil.rmtree(d, ignore_errors=True)

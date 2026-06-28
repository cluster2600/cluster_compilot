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
        cc = ["clang", "-O3", "-std=c11", "-Xclang", "-fopenmp", src, "-o", binp, "-lm"]
        if omp:
            cc[5:5] = [f"-I{omp}/include"]
            cc += [f"-L{omp}/lib", "-lomp"]
        cp = subprocess.run(cc, capture_output=True, text=True, timeout=timeout)
        if cp.returncode != 0:
            return {"ok": False, "error": "compile_error", "detail": cp.stderr[-800:]}
        env = dict(os.environ)
        if omp:
            env["DYLD_LIBRARY_PATH"] = f"{omp}/lib:" + env.get("DYLD_LIBRARY_PATH", "")
        if threads:
            env["OMP_NUM_THREADS"] = str(threads)
        with _RUN_LOCK:                       # serialize timed runs; see _RUN_LOCK
            rp = subprocess.run([binp], capture_output=True, text=True, timeout=timeout, env=env)
        if rp.returncode != 0:
            return {"ok": False, "error": "runtime_error", "detail": rp.stderr[-800:]}
        t = re.search(r"TIME\s+([0-9.eE+-]+)", rp.stdout)
        c = re.search(r"CHECKSUM\s+([0-9.eE+-]+)", rp.stdout)
        if not (t and c):
            return {"ok": False, "error": "no_output", "detail": rp.stdout[-400:]}
        return {"ok": True, "time": float(t.group(1)), "checksum": float(c.group(1))}
    finally:
        shutil.rmtree(d, ignore_errors=True)

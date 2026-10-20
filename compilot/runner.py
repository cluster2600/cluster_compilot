"""Compile a generated C program with clang + OpenMP, run it, parse TIME/CHECKSUM.

This is the *measurement* half of the environment: real wall-clock speedup, which
the paper gets from executing Tiramisu-generated code. We execute clang-generated
code instead — the measured number is equally real.

A single run of a parallel kernel is noisy (identical schedules vary 2-3x run to
run on a busy multicore), so a search that keeps the max of many single runs
over-reports its winner. confirm_speedup re-measures a chosen candidate:
  - default: `rounds` fresh runs of the candidate, median vs the cached baseline.
    That removes the selection bias (the baseline is a shared denominator, so its
    noise rescales every candidate alike) and is cheap, since winners run fast.
  - interleaved: ab_speedup — baseline and candidate alternate (ABBA...) and the
    median ratio is kept, which also cancels drift and baseline noise. Costs a
    baseline run per round; for benchmarks, not the search loop.
"""
import os
import re
import shutil
import statistics
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

CONFIRM_ROUNDS = 5   # baseline/candidate run pairs per ab_speedup


def _libomp_prefix():
    global _OMP
    if _OMP is None:
        try:
            _OMP = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
        except Exception:
            _OMP = ""
    return _OMP


def _compile(c_code, d, timeout):
    """Compile c_code into directory d. Return (binary_path, None) or (None, error dict)."""
    omp = _libomp_prefix()
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
        return None, {"ok": False, "error": "timeout", "detail": "compile timed out"}
    if cp.returncode != 0:
        return None, {"ok": False, "error": "compile_error", "detail": cp.stderr[-800:]}
    return binp, None


def _run(binp, threads, timeout):
    """Run a compiled kernel once (serialized by _RUN_LOCK). Return the result dict."""
    omp = _libomp_prefix()
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


def compile_and_run(c_code, threads=None, timeout=120):
    """Return dict(ok, time, checksum, error). One compile + one run."""
    d = tempfile.mkdtemp(prefix="compilot_")
    try:
        binp, err = _compile(c_code, d, timeout)
        return err or _run(binp, threads, timeout)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def checksums_match(a, b):
    """The correctness gate every environment applies against the baseline."""
    return abs(a - b) <= 1e-6 * max(1.0, abs(b))


def rerun(c_code, runs=CONFIRM_ROUNDS, threads=None, timeout=120):
    """Compile once, run `runs` times. dict(ok, time=median, times, checksum) or an
    error dict; every run must print the same checksum."""
    d = tempfile.mkdtemp(prefix="compilot_rr_")
    try:
        binp, err = _compile(c_code, d, timeout)
        if err:
            return err
        times, csum = [], None
        for _ in range(runs):
            r = _run(binp, threads, timeout)
            if not r["ok"]:
                return r
            if csum is not None and not checksums_match(r["checksum"], csum):
                return {"ok": False, "error": "incorrect", "detail": "checksum differs between runs (race?)"}
            csum = r["checksum"]
            times.append(r["time"])
        return {"ok": True, "time": statistics.median(times), "times": times, "checksum": csum}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def confirm_speedup(baseline, base_code, cand_code, rounds=CONFIRM_ROUNDS, interleaved=False):
    """Re-measured speedup of cand over the baseline (see module docstring), or None
    if it fails or its checksum doesn't match. `baseline` is the env's cached
    baseline result; `base_code` is only compiled when interleaved."""
    if interleaved:
        r = ab_speedup(base_code, cand_code, rounds)
        return r["speedup"] if r["ok"] else None
    r = rerun(cand_code, rounds)
    if not r["ok"] or not checksums_match(r["checksum"], baseline["checksum"]):
        return None
    return baseline["time"] / r["time"]


def ab_speedup(base_code, cand_code, rounds=CONFIRM_ROUNDS, threads=None, timeout=120):
    """Re-measure cand vs base: compile both once, then run them in alternating
    order (ABBA...) for `rounds` pairs, so drift (thermal, frequency, background
    load) hits both alike. Returns dict(ok, speedup=median(base/cand), ratios,
    base_time, time) or an error dict; checks every candidate checksum."""
    d = tempfile.mkdtemp(prefix="compilot_ab_")
    try:
        os.mkdir(os.path.join(d, "a"))
        os.mkdir(os.path.join(d, "b"))
        base_bin, err = _compile(base_code, os.path.join(d, "a"), timeout)
        if err:
            return err
        cand_bin, err = _compile(cand_code, os.path.join(d, "b"), timeout)
        if err:
            return err
        ratios, bts, cts = [], [], []
        for r in range(rounds):
            order = (base_bin, cand_bin) if r % 2 == 0 else (cand_bin, base_bin)
            res = {p: _run(p, threads, timeout) for p in order}
            b, c = res[base_bin], res[cand_bin]
            for x in (b, c):
                if not x["ok"]:
                    return x
            if not checksums_match(c["checksum"], b["checksum"]):
                return {"ok": False, "error": "incorrect",
                        "detail": f"checksum {c['checksum']:.6e} != baseline {b['checksum']:.6e}"}
            ratios.append(b["time"] / c["time"])
            bts.append(b["time"])
            cts.append(c["time"])
        return {"ok": True, "speedup": statistics.median(ratios), "ratios": ratios,
                "base_time": statistics.median(bts), "time": statistics.median(cts)}
    finally:
        shutil.rmtree(d, ignore_errors=True)

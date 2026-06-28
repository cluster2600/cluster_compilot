"""ISL backend: the environment the agent talks to.

evaluate(schedule) does what Tiramisu does for ComPilot, via ISL + clang:
  1. parse the schedule
  2. prove legality with polyhedral dependence analysis (ISL)
  3. check each requested parallel() level is actually parallelizable
  4. if legal: generate C, compile, run, and return the measured speedup
     (cross-checking the output checksum against the untransformed baseline —
      a second, independent correctness guard alongside polyhedral legality)

Returns a Result whose `status` is one of the paper's feedback categories.
"""
import threading
from dataclasses import dataclass
from . import schedule as _schedule
from . import codegen as _codegen
from . import runner as _runner
from .polyhedral import dependences, is_legal, is_parallel
from .scheduler import build_theta

# islpy builds every object in its process-global DEFAULT_CONTEXT, and ISL is not
# thread-safe within one context. evaluate() runs concurrently when the agent
# fans out (parallel best-of-k / multi-candidate turns), so the polyhedral
# section is serialized by this lock. The slow part — clang compile + run, a
# subprocess that releases the GIL — stays OUTSIDE the lock and runs in parallel,
# which is where the wall-clock speedup actually comes from.
_ISL_LOCK = threading.Lock()

# transforms the C codegen can currently emit (legality covers all 9; execution
# of skew/reverse/fuse/shift is added with their codegen).
_EXECUTABLE = {"interchange", "reorder", "tile", "tile2d", "tile3d", "parallel", "unroll", "reverse", "skew"}


def environment(name, size="LARGE"):
    """Build an Environment for a registered kernel name at a PolyBench size class."""
    from .kernels import sized_kernel
    ek, pk = sized_kernel(name, size)
    return Environment(ek, pk)


@dataclass
class Result:
    status: str            # success | illegal | parallel_illegal | invalid | compile_error
                           # | runtime_error | incorrect | unsupported
    speedup: float = None
    detail: str = ""
    schedule: str = ""


class Environment:
    """Holds a kernel pair (exec spec + polyhedral spec) and its cached baseline."""

    def __init__(self, exec_kernel, poly_kernel):
        self.ek = exec_kernel
        self.pk = poly_kernel
        self.D = dependences(poly_kernel)
        self._baseline = None
        self._baseline_lock = threading.Lock()
        self._cache = {}                          # schedule key -> Result (shared across the search)
        self._cache_lock = threading.Lock()

    def baseline(self):
        # Compiled once and shared across concurrent runs. Pre-warm by calling this
        # before fanning out; the double-checked lock keeps it safe (and single
        # compile) even if first touched mid-fan-out. A dedicated lock means the
        # baseline subprocess never blocks other threads' legality checks.
        if self._baseline is None:
            with self._baseline_lock:
                if self._baseline is None:
                    r = _runner.compile_and_run(_codegen.generate_c(self.ek, ""))
                    if not r["ok"]:
                        raise RuntimeError(f"baseline failed: {r}")
                    self._baseline = r
        return self._baseline

    def evaluate(self, schedule_text) -> Result:
        """Memoized: the same schedule (whitespace-insensitive) is compiled and timed
        once, then its Result is reused across the whole parallel search. Skips the
        redundant clang compile + timed run when runs/candidates re-propose a schedule."""
        key = "".join(schedule_text.split())
        with self._cache_lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._evaluate(schedule_text)
        with self._cache_lock:
            # ponytail: a rare duplicate compile if two threads miss at once; setdefault
            # makes them converge on one stored Result rather than locking the whole eval.
            return self._cache.setdefault(key, result)

    def _evaluate(self, schedule_text) -> Result:
        try:
            ops = _schedule.parse(schedule_text)
        except ValueError as e:
            return Result("invalid", detail=str(e), schedule=schedule_text)

        # --- polyhedral legality (all 9 primitives) ---
        # Serialized: build_theta + is_legal + is_parallel all operate on islpy
        # objects in the shared DEFAULT_CONTEXT, which is not thread-safe. This is
        # the fast part (microseconds-milliseconds); the lock is released before
        # the expensive compile+run below so those proceed in parallel.
        with _ISL_LOCK:
            try:
                theta, labels, par, unroll = build_theta(self.pk, ops)
            except ValueError as e:
                return Result("invalid", detail=str(e), schedule=schedule_text)
            legal, viol = is_legal(self.D, theta)
            if not legal:
                return Result("illegal", detail=f"violates dependences: {viol}", schedule=schedule_text)
            for lbl, lvl in par:
                if not is_parallel(self.D, theta, lvl):
                    return Result("parallel_illegal",
                                  detail=f"loop {lbl} carries a dependence; cannot parallelize",
                                  schedule=schedule_text)

        # --- execution / measurement ---
        used = {op for op, _ in ops}
        if not used <= _EXECUTABLE:
            return Result("unsupported", detail=f"legal, but codegen lacks {used - _EXECUTABLE}",
                          schedule=schedule_text)
        base = self.baseline()
        r = _runner.compile_and_run(_codegen.generate_c(self.ek, schedule_text))
        if not r["ok"]:
            return Result(r["error"], detail=r.get("detail", ""), schedule=schedule_text)
        ref = base["checksum"]
        if abs(r["checksum"] - ref) > 1e-6 * max(1.0, abs(ref)):
            return Result("incorrect",
                          detail=f"checksum {r['checksum']:.6e} != baseline {ref:.6e} "
                                 f"(ISL said legal — codegen bug)", schedule=schedule_text)
        return Result("success", speedup=base["time"] / r["time"],
                      detail=f"{base['time']:.4f}s -> {r['time']:.4f}s", schedule=schedule_text)

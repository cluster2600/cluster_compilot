"""ISL backend: the environment the agent talks to.

evaluate(schedule) does what Tiramisu does for ComPilot, via ISL + clang:
  1. parse the schedule
  2. prove legality with polyhedral dependence analysis (ISL)
  3. check each requested parallel() level is actually parallelizable
  4. if legal: generate C from that same proven schedule (ISL AST generation),
     compile, run, and return the measured speedup (cross-checking the output
     checksum against the untransformed baseline — a second, independent
     correctness guard alongside polyhedral legality)

confirm(schedule) re-measures a schedule (median of fresh runs; or interleaved
A/B rounds against the baseline) — the agent uses it on each new best, so the
reported best is not just the luckiest of many noisy single runs.

Returns a Result whose `status` is one of the paper's feedback categories.
"""
import islpy as isl
import threading
from dataclasses import dataclass
from . import schedule as _schedule
from . import codegen as _codegen
from . import runner as _runner
from .polyhedral import ISL_LOCK, dependences, is_legal, is_parallel
from .scheduler import build_theta

# Kept under its old name for callers/docs; the one lock lives in polyhedral.
_ISL_LOCK = ISL_LOCK

# Errors a malformed schedule can raise anywhere in parse -> θ' -> AST codegen
# (too few args, bad loop name, non-numeric factor, ISL rejecting an expression).
SCHEDULE_ERRORS = (ValueError, IndexError, KeyError, isl.Error)


def environment(name, size="LARGE"):
    """Build an Environment for a registered kernel name at a size class.

    Imperfect-nest kernels (solvers) get an ImperfectEnvironment, which exposes the
    same .pk/.baseline()/.evaluate() interface, so the single-statement agent
    dialogue, prompt builder, and MCP/TUI single-schedule paths work unchanged."""
    from .kernels import sized_kernel, IMPERFECT_REGISTRY
    if name in IMPERFECT_REGISTRY:
        from .imperfect import ImperfectEnvironment
        return ImperfectEnvironment(sized_kernel(name, size))
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
        with ISL_LOCK:
            self.D = dependences(poly_kernel)
        self._baseline = None
        self._baseline_lock = threading.Lock()
        self._cache = {}                          # schedule key -> Result (shared across the search)
        self._cache_lock = threading.Lock()

    def baseline_program(self):
        return _codegen.generate_c(self.ek, "")

    def baseline(self):
        # Compiled once and shared across concurrent runs. Pre-warm by calling this
        # before fanning out; the double-checked lock keeps it safe (and single
        # compile) even if first touched mid-fan-out. A dedicated lock means the
        # baseline subprocess never blocks other threads' legality checks.
        if self._baseline is None:
            with self._baseline_lock:
                if self._baseline is None:
                    r = _runner.compile_and_run(self.baseline_program())
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

    def _check(self, schedule_text):
        """Parse, prove legality, generate C. Returns (program, None) or (None, Result).

        Serialized: θ' construction, legality, parallelism and AST codegen all use
        islpy objects in the shared context, which is not thread-safe. This is the
        fast part (milliseconds); the lock is released before the expensive
        compile+run, so those proceed in parallel."""
        with ISL_LOCK:
            try:
                ops = _schedule.parse(schedule_text)
                theta, labels, par, unroll = build_theta(self.pk, ops)
            except SCHEDULE_ERRORS as e:
                return None, Result("invalid", detail=str(e), schedule=schedule_text)
            legal, viol = is_legal(self.D, theta)
            if not legal:
                return None, Result("illegal", detail=f"violates dependences: {viol}",
                                    schedule=schedule_text)
            for lbl, lvl in par:
                if not is_parallel(self.D, theta, lvl):
                    return None, Result("parallel_illegal",
                                        detail=f"loop {lbl} carries a dependence; cannot parallelize",
                                        schedule=schedule_text)
            try:
                nest = _codegen.isl_nest(self.pk, self.ek.sizes, self.ek.body, theta, par, unroll)
            except SCHEDULE_ERRORS as e:
                return None, Result("invalid", detail=str(e), schedule=schedule_text)
        return _codegen.program(self.ek, nest), None

    def _evaluate(self, schedule_text) -> Result:
        program, failed = self._check(schedule_text)
        if failed:
            return failed
        base = self.baseline()
        r = _runner.compile_and_run(program)
        if not r["ok"]:
            return Result(r["error"], detail=r.get("detail", ""), schedule=schedule_text)
        ref = base["checksum"]
        if not _runner.checksums_match(r["checksum"], ref):
            return Result("incorrect",
                          detail=f"checksum {r['checksum']:.6e} != baseline {ref:.6e} "
                                 f"(ISL said legal — polyhedral spec and C body disagree)",
                          schedule=schedule_text)
        return Result("success", speedup=base["time"] / r["time"],
                      detail=f"{base['time']:.4f}s -> {r['time']:.4f}s", schedule=schedule_text)

    def confirm(self, schedule_text, rounds=_runner.CONFIRM_ROUNDS, interleaved=False):
        """Re-measured speedup of `schedule_text` (runner.confirm_speedup: median of
        `rounds` fresh runs, or interleaved A/B rounds), or None if it is not a
        legal, correct schedule."""
        program, failed = self._check(schedule_text)
        if failed:
            return None
        return _runner.confirm_speedup(self.baseline(), self.baseline_program(), program,
                                       rounds, interleaved)

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
from dataclasses import dataclass
from . import schedule as _schedule
from . import codegen as _codegen
from . import runner as _runner
from .polyhedral import dependences, is_legal, is_parallel
from .scheduler import build_theta

# transforms the C codegen can currently emit (legality covers all 9; execution
# of skew/reverse/fuse/shift is added with their codegen).
_EXECUTABLE = {"interchange", "reorder", "tile", "tile2d", "tile3d", "parallel", "unroll"}


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

    def baseline(self):
        if self._baseline is None:
            r = _runner.compile_and_run(_codegen.generate_c(self.ek, ""))
            if not r["ok"]:
                raise RuntimeError(f"baseline failed: {r}")
            self._baseline = r
        return self._baseline

    def evaluate(self, schedule_text) -> Result:
        try:
            ops = _schedule.parse(schedule_text)
        except ValueError as e:
            return Result("invalid", detail=str(e), schedule=schedule_text)

        # --- polyhedral legality (all 9 primitives) ---
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

"""Measurement: a new best is re-measured before it is kept.

Single timed runs of a parallel kernel vary 2-3x, so a search that keeps the max
of many raw runs reports its luckiest sample (winner's curse). The fix:
  - runner.rerun / confirm_speedup: median of fresh runs of the chosen schedule
  - runner.ab_speedup (interleaved=True): baseline and candidate alternate, median
    ratio; used by bench.py / baselines.py; both check checksums
  - every environment has confirm(schedule, interleaved=False)
  - the agent keeps the CONFIRMED number for a new best, at most one confirm/turn

    python3 -m tests.test_confirm
"""
from dataclasses import replace

from compilot import codegen, runner
from compilot.agent import run_dialogue
from compilot.backend_isl import Result, environment
from compilot.llm import MockClient
from compilot.multikernel import MultiEnvironment
from compilot.kernels import sized_kernel


def test_ab_speedup_median_and_checksum_gate():
    env = environment("gemm", "SMALL")
    base = env.baseline_program()
    cand = codegen.generate_c(env.ek, "reorder(i, k, j)", env.pk)
    r = runner.ab_speedup(base, cand, rounds=3)
    assert r["ok"] and len(r["ratios"]) == 3 and r["speedup"] > 0, r
    wrong = codegen.generate_c(replace(env.ek, body=env.ek.body.replace("+=", "-=")), "")
    bad = runner.ab_speedup(base, wrong, rounds=1)
    assert not bad["ok"] and bad["error"] == "incorrect", bad
    assert runner.confirm_speedup(env.baseline(), base, wrong, rounds=1) is None   # rerun path
    rr = runner.rerun(cand, runs=3)
    assert rr["ok"] and len(rr["times"]) == 3, rr
    print(f"OK ab_speedup: median of {r['ratios']} -> {r['speedup']:.2f}x; wrong result -> incorrect")


def test_confirm_on_every_environment():
    gemm = environment("gemm", "SMALL")
    assert gemm.confirm("reorder(i, k, j)", rounds=2) > 0
    assert gemm.confirm("reorder(i, k, j)", rounds=2, interleaved=True) > 0
    assert gemm.confirm("reverse(k)", rounds=1) is None                  # illegal -> no number
    mm = MultiEnvironment(sized_kernel("2mm", "SMALL"))
    assert mm.confirm(["parallel(i)", ""], rounds=1) > 0
    assert mm.confirm(["parallel(i)", ""], rounds=1, interleaved=True) > 0
    assert environment("lu", "SMALL").confirm("parallel(i)", rounds=1) > 0  # imperfect nest
    print("OK confirm() on single / multi / imperfect environments")


class _NoisyEnv:
    """Raw single runs over-report schedules with parallel(); confirm tells the truth."""

    def __init__(self):
        real = environment("gemm", "SMALL")
        self.ek, self.pk = real.ek, real.pk
        self.confirmed = []

    def baseline(self):
        return {"time": 1.0, "checksum": 0.0}

    def evaluate(self, s):
        return Result("success", speedup=50.0 if "parallel" in s else 3.0, schedule=s)

    def confirm(self, s):
        self.confirmed.append(s)
        return 20.0 if "parallel" in s else 3.0


def test_agent_keeps_confirmed_best():
    env = _NoisyEnv()
    sp, sched, trace = run_dialogue(env, MockClient(), max_iters=6, verbose=False)
    assert sp == 20.0, sp                                    # not the raw 50x
    assert "parallel(i_t)" in sched and "parallel(k)" not in sched, sched
    assert max(t[2] for t in trace) == 50.0                  # the trace keeps raw numbers
    print(f"OK agent best = confirmed {sp}x (raw max was 50x); {len(env.confirmed)} confirms")


class _ThreeAtOnce:
    in_tokens = out_tokens = 0

    def __init__(self):
        self.n = 0

    def chat(self, system, messages):
        self.n += 1
        if self.n > 1:
            return "<schedule>no_further_transformations</schedule>"
        return ("<schedule>reorder(i, k, j)</schedule><schedule>reorder(i, k, j)\nparallel(i)</schedule>"
                "<schedule>interchange(i, j)</schedule>")


def test_one_confirm_per_turn():
    env = _NoisyEnv()
    sp, _, _ = run_dialogue(env, _ThreeAtOnce(), max_iters=3, verbose=False, candidates_per_turn=3)
    assert env.confirmed == ["reorder(i, k, j)\nparallel(i)"], env.confirmed
    assert sp == 20.0, sp
    print("OK multi-candidate turn confirms only its top raw candidate")


def test_measurement_reports_threads_and_busy_hosts():
    import os
    env = environment("gemm", "SMALL")
    r = runner.compile_and_run(codegen.generate_c(env.ek, "parallel(i)", env.pk))
    assert r["ok"] and r["threads"] and r["threads"] >= 1, r      # programs print THREADS
    orig = os.getloadavg
    try:
        os.getloadavg = lambda: (1000.0, 0.0, 0.0)
        assert "busy" in runner.busy_warning()
        os.getloadavg = lambda: (0.0, 0.0, 0.0)
        assert runner.busy_warning() is None
    finally:
        os.getloadavg = orig
    # thread pinning is opt-in and never overrides the user's own OpenMP settings
    assert "OMP_PROC_BIND" not in runner.omp_env({})
    assert runner.omp_env({"COMPILOT_OMP_BIND": "1"})["OMP_PLACES"] == "cores"
    assert runner.omp_env({"COMPILOT_OMP_BIND": "1", "OMP_PROC_BIND": "spread"})["OMP_PROC_BIND"] == "spread"
    print(f"OK runs report threads ({r['threads']}); busy-host warning; pinning opt-in")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
    print(f"\ntest_confirm: all {len(_TESTS)} checks passed")

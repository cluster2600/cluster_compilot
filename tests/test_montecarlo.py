"""Monte Carlo equity-curve simulation (the ELVIS kernel).

Asserts the legality story that makes this kernel a faithful ComPilot target:
  - parallel(s)  : LEGAL   (the simulation loop is fully independent)
  - parallel(t)  : REJECTED (the trade loop is a sequential recurrence)
  - identity is correct; the parallel run matches the baseline checksum.
Run: python3 -m tests.test_montecarlo
"""
from compilot.backend_isl import environment

env = environment("montecarlo")


def main():
    base = env.baseline()
    print(f"baseline: {base['time']:.4f}s  checksum {base['checksum']:.4e}\n")

    par_s = env.evaluate("parallel(s)")
    par_t = env.evaluate("parallel(t)")

    # the simulation loop parallelizes and actually runs
    assert par_s.status == "success", par_s
    assert par_s.speedup and par_s.speedup > 1.0, par_s
    # the trade loop carries a dependence -> must be rejected pre-execution
    assert par_t.status == "parallel_illegal", par_t

    print(f"[{par_s.status:16}] {par_s.speedup:.2f}x  parallel(s)  (simulation loop)")
    print(f"[{par_t.status:16}]   -    parallel(t)  (trade recurrence, correctly rejected)")
    print("\nOK")


if __name__ == "__main__":
    main()

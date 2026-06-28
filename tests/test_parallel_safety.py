"""Concurrency safety: parallel env.evaluate() must agree with serial.

islpy builds objects in a process-global, non-thread-safe context, so evaluate()
guards its polyhedral section with a lock. This hammers a shared environment with
many concurrent evaluate() calls and asserts every schedule's legality verdict
matches the serial reference (speedups jitter, so we compare *status*). A missing
or broken lock shows up here as a crash, a hang, or a mismatched status.
"""
from concurrent.futures import ThreadPoolExecutor

from compilot.backend_isl import environment

SCHEDULES = [
    "",                                                   # identity        -> success
    "reorder(i, k, j)",                                   # legal reorder   -> success
    "reorder(i, k, j)\ntile2d(i, j, 64, 64)\nparallel(i_t)",  # tiled+parallel -> success
    "interchange(i, j)",                                  # legal swap      -> success
    "tile(i, 32)\nparallel(i_t)",                         # tiled+parallel  -> success
    "reverse(k)",                                         # reduction rev   -> illegal
    "parallel(k)",                                        # reduction loop  -> parallel_illegal
    "bogus(i)",                                           # parse/unknown   -> invalid
]


def main():
    env = environment("gemm")
    env.baseline()                                        # pre-warm before fan-out

    serial = {s: env.evaluate(s).status for s in SCHEDULES}

    # Hammer: many rounds, each evaluating every schedule concurrently with high
    # worker count so multiple threads sit in the polyhedral section at once.
    work = SCHEDULES * 8
    mismatches = 0
    for rnd in range(6):
        with ThreadPoolExecutor(max_workers=16) as ex:
            out = list(ex.map(lambda s: (s, env.evaluate(s).status), work))
        for s, status in out:
            if status != serial[s]:
                mismatches += 1
                print(f"  MISMATCH round {rnd}: {s!r} -> {status} (serial {serial[s]})")

    for s in SCHEDULES:
        print(f"[{serial[s]:16}] {s.splitlines()[0] if s else '(identity)'}")
    assert mismatches == 0, f"{mismatches} parallel/serial status mismatches"
    print(f"\nOK: {len(work) * 6} concurrent evaluate() calls, 0 mismatches, "
          f"verdicts identical to serial.")


if __name__ == "__main__":
    main()

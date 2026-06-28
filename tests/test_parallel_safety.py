"""Concurrency safety: parallel env.evaluate() must agree with serial.

islpy builds objects in a process-global, non-thread-safe context, so evaluate()
guards its polyhedral section with a lock. This hammers a shared environment with
many concurrent calls and asserts every schedule's legality verdict matches the
serial reference (speedups jitter, so we compare *status*). A missing or broken
lock shows up here as a crash, a hang, or a mismatched status.

evaluate() now memoizes by schedule, so the hammer calls the uncached _evaluate
directly — otherwise every call past the first would hit the cache and never
re-enter the locked section. A separate check covers the cache itself.
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

    serial = {s: env._evaluate(s).status for s in SCHEDULES}    # uncached reference

    # Hammer: many rounds, each evaluating every schedule concurrently with high
    # worker count so multiple threads sit in the polyhedral section at once. Call
    # the uncached _evaluate so every call re-enters the locked section (the public
    # evaluate would memoize after the first hit and stop stressing the lock).
    work = SCHEDULES * 8
    mismatches = 0
    for rnd in range(6):
        with ThreadPoolExecutor(max_workers=16) as ex:
            out = list(ex.map(lambda s: (s, env._evaluate(s).status), work))
        for s, status in out:
            if status != serial[s]:
                mismatches += 1
                print(f"  MISMATCH round {rnd}: {s!r} -> {status} (serial {serial[s]})")

    for s in SCHEDULES:
        print(f"[{serial[s]:16}] {s.splitlines()[0] if s else '(identity)'}")
    assert mismatches == 0, f"{mismatches} parallel/serial status mismatches"
    print(f"\nOK: {len(work) * 6} concurrent evaluate() calls, 0 mismatches, "
          f"verdicts identical to serial.")

    # Memoization: the public evaluate() returns one shared Result per schedule,
    # whitespace-insensitive, so duplicate proposals skip the clang compile + run.
    a = env.evaluate("reorder(i, k, j)")
    b = env.evaluate("reorder(i,k,j)\n")
    assert a is b, "memoized evaluate() should return the same Result for the same schedule"
    assert a.status == env._evaluate("reorder(i,k,j)").status, "cached verdict diverged from uncached"
    print(f"OK: evaluate() memoized (whitespace-insensitive); {len(env._cache)} unique schedules cached.")


if __name__ == "__main__":
    main()

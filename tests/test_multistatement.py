"""Prove the multi-statement legality model on a producer->consumer kernel.

  S0[i]:  B[i] = A[i] + 1        (producer, writes B)
  S1[i]:  C[i] = B[i] * 2        (consumer, reads B from S0)

Dependence: S0[i] -> S1[i] (RAW on B). Therefore:
  - fused        for i: {S0; S1}        legal
  - distributed  for i: S0;  for i: S1   legal
  - reordered    for i: {S1; S0}        ILLEGAL (consumer before producer)
Run: python3 -m tests.test_multistatement
"""
from compilot.polyhedral_multi import Statement, MultiKernel, legal

PC = MultiKernel(
    name="prodcons",
    params=["N"],
    stmts=[
        Statement("S0", ["i"], "0<=i<N", writes=[("B", "i")], reads=[("A", "i")]),
        Statement("S1", ["i"], "0<=i<N", writes=[("C", "i")], reads=[("B", "i")]),
    ],
    # original = fused: same i, S0 (beta 0) before S1 (beta 1)
    sched_map={"S0": ["i", "0"], "S1": ["i", "1"]},
)

CASES = [
    ("fused        {S0;S1}", {"S0": ["i", "0"], "S1": ["i", "1"]}, True),
    ("distributed  S0;;S1",  {"S0": ["0", "i"], "S1": ["1", "i"]}, True),
    ("reordered    {S1;S0}", {"S0": ["i", "1"], "S1": ["i", "0"]}, False),
]

if __name__ == "__main__":
    for name, mapping, expect in CASES:
        got = legal(PC, mapping)
        status = "OK " if got == expect else "FAIL"
        print(f"[{status}] {name:22} legal={got} (want {expect})")
        assert got == expect, f"{name}: {got} != {expect}"
    print("\nMulti-statement legality model validated.")

"""Prove the multi-statement legality model on a producer->consumer kernel.

  S0[i]:  B[i] = A[i] + 1        (producer, writes B)
  S1[i]:  C[i] = B[i] * 2        (consumer, reads B from S0)

Dependence: S0[i] -> S1[i] (RAW on B). Therefore:
  - fused        for i: {S0; S1}        legal
  - distributed  for i: S0;  for i: S1   legal
  - reordered    for i: {S1; S0}        ILLEGAL (consumer before producer)

The program is a union.UnionProgram, the model fusion and the imperfect nests
run on; each case is a 2d+1 schedule [b0, i, b1].
Run: python3 -m tests.test_multistatement
"""
from compilot.polyhedral import dependences, is_legal
from compilot.union import UnionProgram, UStmt


def _stmt(name, body, writes, reads, time):
    return UStmt(name, ["i"], "0<=i<N", reads=[(reads, "i")], writes=[(writes, "i")],
                 body=body, time=time, labels=["i"])


# original = fused: same i, S0 (beta 0) before S1 (beta 1)
PC = UnionProgram([_stmt("S0", "B[i] = A[i] + 1;", "B", "A", ["0", "i", "0"]),
                   _stmt("S1", "C[i] = B[i] * 2;", "C", "B", ["0", "i", "1"])],
                  params=["N"])

CASES = [
    ("fused        {S0;S1}", [["0", "i", "0"], ["0", "i", "1"]], True),
    ("distributed  S0;;S1",  [["0", "i", "0"], ["1", "i", "0"]], True),
    ("reordered    {S1;S0}", [["0", "i", "1"], ["0", "i", "0"]], False),
]

if __name__ == "__main__":
    D = dependences(PC)
    for name, times, expect in CASES:
        got = is_legal(D, PC.schedule(times))[0]
        status = "OK " if got == expect else "FAIL"
        print(f"[{status}] {name:22} legal={got} (want {expect})")
        assert got == expect, f"{name}: {got} != {expect}"
    print("\nMulti-statement legality model validated.")

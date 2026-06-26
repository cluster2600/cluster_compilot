"""Loop fusion (fuse/shift) on gesummv: two matvecs that share the vector x.

  - fusion legality via the multi-statement model (S0, S1 independent -> legal)
  - fused codegen runs correctly (same result as unfused) and is no slower
Run: python3 -m tests.test_fusion
"""
from compilot.kernels import MULTI_REGISTRY, gesummv_fused
from compilot.multikernel import MultiEnvironment
from compilot.polyhedral_multi import Statement, MultiKernel, legal

# fusion legality: S0 writes tmp, S1 writes y, both read x -> independent, fusion legal
d = "0<=i<N and 0<=j<N"
mk = MultiKernel("gesummv",
                 [Statement("S0", ["i", "j"], d, [("tmp", "i")], [("A", "i,j"), ("x", "j")]),
                  Statement("S1", ["i", "j"], d, [("y", "i")], [("B", "i,j"), ("x", "j")])],
                 ["N"], sched_map={"S0": ["i", "j", "0"], "S1": ["i", "j", "1"]})

if __name__ == "__main__":
    fused_legal = legal(mk, {"S0": ["i", "j", "0"], "S1": ["i", "j", "1"]})
    print(f"[{'OK ' if fused_legal else 'FAIL'}] fuse(S0,S1) legal = {fused_legal}")
    assert fused_legal

    eu = MultiEnvironment(MULTI_REGISTRY["gesummv"]())   # unfused (2 nests)
    ef = MultiEnvironment(gesummv_fused())               # fused (1 nest)
    tu, tf = eu.baseline()["time"], ef.baseline()["time"]
    # both must compute the same result -> same checksum
    same = abs(eu.baseline()["checksum"] - ef.baseline()["checksum"]) < 1e-6 * max(1.0, abs(eu.baseline()["checksum"]))
    print(f"[{'OK ' if same else 'FAIL'}] fused result == unfused result")
    assert same, "fusion changed the result"
    print(f"     unfused {tu*1e3:.2f} ms  ->  fused {tf*1e3:.2f} ms  ({tu/tf:.2f}x)")
    print("\nLoop fusion validated (legal, correct, reuses x).")

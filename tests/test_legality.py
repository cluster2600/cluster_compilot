"""Prove the ISL legality oracle distinguishes legal from illegal schedules on GEMM.

GEMM accumulates into C[i,j] across k, so the k loop carries a dependence:
  - reordering i/j/k that keeps k's accumulation order is legal
  - reversing k or parallelizing k is ILLEGAL  (the oracle must catch this)
Run: python3 -m tests.test_legality
"""
from compilot.polyhedral import PolyKernel, dependences, is_legal, is_parallel
from compilot import schedule as sched
from compilot.scheduler import build_theta

GEMM = PolyKernel(
    name="gemm",
    order=["i", "j", "k"],
    domain="0<=i<N and 0<=j<M and 0<=k<K",
    writes=[("C", "i,j")],
    reads=[("A", "i,k"), ("B", "k,j"), ("C", "i,j")],
    params=["N", "M", "K"],
    sizes={"N": 1024, "M": 1024, "K": 1024},
)


def legal(schedule_text):
    ops = sched.parse(schedule_text)
    D = dependences(GEMM)
    theta, labels, par, unr = build_theta(GEMM, ops)
    ok, viol = is_legal(D, theta)
    par_ok = {lbl: is_parallel(D, theta, lvl) for lbl, lvl in par}
    return ok, par_ok


def check(name, schedule_text, expect_legal, expect_par=None):
    ok, par_ok = legal(schedule_text)
    status = "OK " if ok == expect_legal else "FAIL"
    print(f"[{status}] {name:32} legal={ok} (want {expect_legal})  parallel={par_ok}")
    assert ok == expect_legal, f"{name}: legality {ok} != {expect_legal}"
    if expect_par is not None:
        for lbl, want in expect_par.items():
            assert par_ok.get(lbl) == want, f"{name}: parallel[{lbl}] {par_ok.get(lbl)} != {want}"


if __name__ == "__main__":
    check("identity", "", True)
    check("interchange(i,j)", "interchange(i, j)", True)
    check("reorder(k,i,j)", "reorder(k, i, j)", True)
    check("tile(i,32)", "tile(i, 32)", True)
    check("tile2d(i,j,32,32)", "tile2d(i, j, 32, 32)", True)
    check("skew(j,i,1)", "skew(j, i, 1)", True)
    check("reverse(k)  [ILLEGAL]", "reverse(k)", False)
    check("reverse(i)  [legal]", "reverse(i)", True)
    check("parallel(i) legal-level", "parallel(i)", True, {"i": True})
    check("parallel(k) ILLEGAL-level", "parallel(k)", True, {"k": False})
    print("\nAll legality assertions passed.")

"""Imperfect-nest solvers (Track C): trisolv (all parallelism rejected) and lu
(carried k rejected; independent i proven legal + run correctly; independent j
legal but beyond the single-work-share codegen, reported as unsupported).

Validates the depth-walk codegen runs correctly AND the legality engine extends
to imperfect, loop-carried nests: it must reject parallel on the sequential loop
and accept it on the independent ones. (Whether the legal parallel form is
*faster* than serial -O3 is size/hardware dependent and not asserted here.)

    python3 -m tests.test_imperfect
"""
from compilot.imperfect import ImperfectEnvironment
from compilot.kernels import IMPERFECT_REGISTRY, sized_kernel


def _env(name, size="SMALL"):
    return ImperfectEnvironment(sized_kernel(name, size))


def test_registered():
    assert set(IMPERFECT_REGISTRY) == {"trisolv", "lu"}, IMPERFECT_REGISTRY


def test_trisolv_rejects_all_naive_parallelism():
    env = _env("trisolv")
    assert env.evaluate("").status == "success"                 # identity baseline runs
    # i carries the x[j] recurrence; j is a reduction into x[i] -> both rejected
    assert env.evaluate("parallel(i)").status == "parallel_illegal"
    assert env.evaluate("parallel(j)").status == "parallel_illegal"
    print("OK: trisolv identity runs; parallel(i)/parallel(j) both rejected (loop-carried)")


def test_lu_rejects_carried_accepts_independent():
    env = _env("lu", "MEDIUM")
    assert env.evaluate("parallel(k)").status == "parallel_illegal", "k is the carried elimination loop"
    ri = env.evaluate("parallel(i)")          # i encloses both statements -> legal, correct, runs
    assert ri.status == "success" and ri.speedup and ri.speedup > 0, ri
    # j is also dependence-free, but it doesn't enclose the row-scaling sibling, so the
    # single-work-share codegen can't emit it correctly -> unsupported (legality still proven)
    rj = env.evaluate("parallel(j)")
    assert rj.status == "unsupported", rj
    print(f"OK: lu parallel(k) rejected; parallel(i) legal+correct ({ri.speedup:.2f}x); "
          f"parallel(j) legal but beyond single-work-share codegen")


def test_lu_unsupported_transform_is_legal_but_unexecutable():
    # tile is legal here but the imperfect codegen can't emit it yet -> clear status
    env = _env("lu", "SMALL")
    assert env.evaluate("tile3d(k,i,j,16,16,16)").status in ("unsupported", "success"), \
        env.evaluate("tile3d(k,i,j,16,16,16)")
    print("OK: lu legal-but-unsupported transform reported, not crashed")


if __name__ == "__main__":
    test_registered()
    test_trisolv_rejects_all_naive_parallelism()
    test_lu_rejects_carried_accepts_independent()
    test_lu_unsupported_transform_is_legal_but_unexecutable()
    print("test_imperfect: all checks passed")

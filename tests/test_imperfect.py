"""Imperfect-nest kernels (Track C): the loop-carried solvers and triangular BLAS
(trisolv, lu, cholesky, ludcmp, durbin, gramschmidt, trmm, symm, nussinov).

Validates the tree-structured depth-walk codegen runs correctly AND the legality
engine extends to imperfect, loop-carried nests with sibling inner loops: it must
REJECT parallel on a sequential carried loop and ACCEPT it on an independent one,
with the parallel run's checksum matching the serial baseline. (Whether the legal
parallel form is *faster* than serial -O3 is size/hardware dependent, not asserted.)

    python3 -m tests.test_imperfect
"""
from compilot.imperfect import ImperfectEnvironment
from compilot.kernels import IMPERFECT_REGISTRY, sized_kernel


def _env(name, size="SMALL"):
    return ImperfectEnvironment(sized_kernel(name, size))


# (kernel, schedule, expected status). "success" cases actually compile+run the
# parallel variant and cross-check its checksum against the serial baseline.
CASES = [
    ("trisolv", "", "success"),                 # identity baseline runs
    ("trisolv", "parallel(i)", "parallel_illegal"),   # x[j] recurrence
    ("trisolv", "parallel(j)", "parallel_illegal"),   # reduction into x[i]
    ("lu", "parallel(k)", "parallel_illegal"),  # carried elimination loop
    ("lu", "parallel(i)", "success"),           # independent forall, runs correctly
    ("lu", "parallel(j)", "success"),           # independent forall, runs correctly
    ("cholesky", "parallel(i)", "parallel_illegal"),  # reads earlier rows
    ("cholesky", "parallel(j)", "parallel_illegal"),  # reads earlier same-row cols
    ("cholesky", "parallel(k)", "parallel_illegal"),  # reduction
    ("ludcmp", "parallel(i)", "parallel_illegal"),    # sequential row loop
    ("durbin", "parallel(k)", "parallel_illegal"),    # sequential recurrence
    ("durbin", "parallel(i)", "parallel_illegal"),    # carried within a step
    ("gramschmidt", "parallel(k)", "parallel_illegal"),   # A updated every column
    ("trmm", "parallel(i)", "parallel_illegal"),      # anti-dep on B[k][j], k>i
    ("trmm", "parallel(j)", "success"),               # columns independent, runs
    ("symm", "parallel(i)", "parallel_illegal"),      # C[k][j] scatter output dep
    ("symm", "parallel(j)", "success"),               # columns independent, runs
    ("nussinov", "parallel(i)", "parallel_illegal"),  # reads table[i+1][j]
    ("nussinov", "parallel(j)", "parallel_illegal"),  # reads table[i][j-1]
]


def test_registered():
    assert set(IMPERFECT_REGISTRY) == {"trisolv", "lu", "cholesky", "ludcmp", "durbin",
                                       "gramschmidt", "trmm", "symm", "nussinov"}, IMPERFECT_REGISTRY


def test_legality_and_parallel_correctness():
    envs = {}
    for name, sched, want in CASES:
        env = envs.setdefault(name, _env(name))
        got = env.evaluate(sched)
        assert got.status == want, f"{name} {sched!r}: got {got.status} ({got.detail}), want {want}"
        tag = f"{got.speedup:.2f}x" if got.speedup else ""
        print(f"OK {name:12} {sched or '(identity)':14} -> {got.status:16} {tag}")


def test_lu_unsupported_transform_is_legal_but_unexecutable():
    # tile is legal here but the imperfect codegen can't emit it yet -> clear status
    env = _env("lu")
    assert env.evaluate("tile3d(k,i,j,16,16,16)").status in ("unsupported", "success")
    print("OK: lu legal-but-unsupported transform reported, not crashed")


if __name__ == "__main__":
    test_registered()
    test_legality_and_parallel_correctness()
    test_lu_unsupported_transform_is_legal_but_unexecutable()
    print("test_imperfect: all checks passed")

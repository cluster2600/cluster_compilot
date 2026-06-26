"""Cross-validate our ISL legality engine against the REAL Tiramisu compiler.

Drives libtiramisu (the exact backend we built) via compilot/backends/tiramisu.py:
emits a Tiramisu C++ program, applies the schedule, runs Tiramisu's own
check_legality_of_function(), and checks the verdict matches our ISL engine.

Skips if libtiramisu isn't built (third_party/, gitignored).
Run: python3 -m tests.test_tiramisu_parity
"""
import os

from compilot import schedule as S
from compilot.polyhedral import dependences, is_legal
from compilot.scheduler import build_theta
from compilot.kernels import GEMM_POLY
from compilot.backends import tiramisu as T

# Directly-comparable transforms (no init/update interleaving differences).
CASES = [
    ("interchange(i, j)", True),
    ("reverse(k)", False),
    ("reverse(i)", True),
    ("tile2d(i, j, 16, 16)", True),
]


def isl_legal(txt):
    theta, _, _, _ = build_theta(GEMM_POLY, S.parse(txt))
    return is_legal(dependences(GEMM_POLY), theta)[0]


if __name__ == "__main__":
    if not os.path.exists(os.path.join(T.BUILD, "libtiramisu.dylib")):
        print("SKIP: libtiramisu not built (third_party/ is gitignored)")
        raise SystemExit(0)
    agree = 0
    for txt, expect in CASES:
        isl = isl_legal(txt)
        tira, info = T.legality(S.parse(txt))
        assert isl == expect, f"ISL {txt}: {isl} != expected {expect}"
        ok = isl == tira
        agree += ok
        print(f"[{'OK ' if ok else 'DIFF'}] {txt:22} ISL={isl}  Tiramisu={tira}")
        assert tira is not None, f"Tiramisu bridge error on {txt}: {info[:200]}"
        assert ok, f"verdict mismatch on {txt}"
    print(f"\nISL vs real Tiramisu: {agree}/{len(CASES)} agree.")

    # Tiramisu's real Halide codegen lowers a scheduled GEMM to an object file
    size, info = T.codegen(S.parse("tile2d(i, j, 32, 32)\nparallel(i1)"))
    print(f"[{'OK ' if size else 'FAIL'}] Tiramisu Halide codegen -> "
          f"{size} bytes object" if size else f"[FAIL] codegen: {info[:200]}")
    assert size and size > 0, f"Tiramisu codegen failed: {info[:200]}"

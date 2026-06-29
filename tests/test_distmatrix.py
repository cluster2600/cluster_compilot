"""Pairwise squared-Euclidean distance matrix (the zvec kernel).

zvec's euclidean_distance_matrix hot loop: out[i][j] = sum_k (m[i][k]-q[j][k])^2.
Compute-bound, so tiling/parallelizing gives a real (non-bandwidth-capped) win.
  - parallel(i) : LEGAL and fast (independent query rows)
  - parallel(k) : REJECTED (k is the reduction dimension)
Run: python3 -m tests.test_distmatrix
"""
from compilot.backend_isl import environment

env = environment("distmatrix")


def main():
    print(f"baseline: {env.baseline()['time']:.4f}s\n")
    par_i = env.evaluate("parallel(i)")
    par_k = env.evaluate("parallel(k)")

    assert par_i.status == "success", par_i
    assert par_i.speedup and par_i.speedup > 0, par_i        # ran; absolute speedup is core/contention dependent (not asserted in CI)
    assert par_k.status == "parallel_illegal", par_k

    print(f"[{par_i.status:16}] {par_i.speedup:.2f}x  parallel(i)  (query rows)")
    print(f"[{par_k.status:16}]   -    parallel(k)  (reduction, correctly rejected)")
    print("\nOK")


if __name__ == "__main__":
    main()

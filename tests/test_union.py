"""Union-of-statements programs: the 2d+1 loop tree, fusion and shifting.

  - tree_times reproduces how the imperfect emitter nests statements
  - fuse(L) runs a statement inside the previous one's loops; the whole union is
    checked, so a fusion that breaks a dependence between statements is rejected
  - shift(L, c) makes the classic jacobi fusion legal; the fused code runs and
    matches the unfused baseline's checksum, also with the fused loop parallel
  - kernels whose per-statement output resets can't be hoisted refuse fusion

    python3 -m tests.test_union
"""
from compilot.kernels import sized_kernel
from compilot.multikernel import MultiEnvironment
from compilot.stencil import StencilEnvironment
from compilot.union import tree_times


def test_tree_times_follow_the_loop_tree():
    I, J = ("i", "0", "N"), ("j", "0", "i")
    t = tree_times([[I], [I, J], [I], [("i", "0", "N", "rev")]])
    assert t == [["0", "i", "0"], ["0", "i", "1", "j", "0"], ["0", "i", "2"], ["1", "-i", "0"]], t
    print("OK tree_times: shared i loop, nested j, then a separate descending loop")


def _multi(name, scheds, size="SMALL"):
    return MultiEnvironment(sized_kernel(name, size)).evaluate(scheds)


def _stencil(name, scheds, size="SMALL"):
    return StencilEnvironment(sized_kernel(name, size)).evaluate(scheds)


def test_fusion_legality_and_execution():
    cases = [
        (_multi, "2mm", ["", "fuse(i)"], "success"),                 # D row i needs tmp row i only
        (_multi, "2mm", ["reorder(i,k,j)\nparallel(i)", "reorder(i,l,j)\nfuse(i)"], "success"),
        (_multi, "2mm", ["", "fuse(j)"], "illegal"),                 # D[i][j] needs all of tmp row i
        (_multi, "3mm", ["", "", "fuse(i)"], "illegal"),            # G needs all of F
        (_multi, "atax", ["", "fuse(i)"], "success"),
        (_stencil, "jacobi1d", ["", "fuse(i)"], "illegal"),         # A[i] = f(B[i+1]) too early
        (_stencil, "jacobi1d", ["", "shift(i, 1)\nfuse(i)"], "success"),
        (_stencil, "jacobi1d", ["parallel(i)", "shift(i, 1)\nfuse(i)"], "parallel_illegal"),
        (_stencil, "jacobi2d", ["", "shift(i, 1)\nfuse(i)"], "success"),
        (_stencil, "jacobi2d", ["parallel(j)", "shift(i, 1)\nfuse(i)"], "success"),
        (_multi, "covariance", ["", "fuse(i)", ""], "invalid"),     # in-place reset can't be hoisted
    ]
    for run, name, scheds, want in cases:
        r = run(name, scheds)
        assert r["status"] == want, f"{name} {scheds}: {r['status']} ({r.get('detail', '')[:120]}), want {want}"
        sp = f"{r['speedup']:.2f}x" if r.get("speedup") else ""
        print(f"OK {name:10} {' | '.join(s.replace(chr(10), '; ') or '-' for s in scheds):54} -> {r['status']} {sp}")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
    print(f"\ntest_union: all {len(_TESTS)} checks passed")

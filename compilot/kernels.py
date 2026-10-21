"""Kernels as schedulable affine loop nests.

Each kernel has two paired specs:
  - Kernel      : execution spec (flat arrays, body, sizes) used by codegen
  - PolyKernel  : polyhedral spec (domain, reads, writes, loop order) used by ISL legality

REGISTRY pairs them by name so the Environment can take a single kernel name.
These are single-statement kernels (output zeroed, then an accumulation nest):
GEMM (C=A·B), SYRK (C=A·Aᵀ), SYR2K (C=A·Bᵀ+B·Aᵀ). Multi-statement PolyBench
kernels arrive with the multi-statement polyhedral model.
"""
from dataclasses import dataclass, field, replace
from .polyhedral import PolyKernel


@dataclass
class Kernel:
    name: str
    sizes: dict
    arrays: dict
    loops: list
    body: str
    output: str
    reduction: set = field(default_factory=set)
    reset: str = "zero"      # per-rep reset of `output`: "zero" or "reinit" (in-place kernels)


# ---- GEMM : C = A * B -----------------------------------------------------
GEMM = Kernel(
    name="gemm", sizes={"N": 512, "M": 512, "K": 512},
    arrays={"A": ("N", "K"), "B": ("K", "M"), "C": ("N", "M")},
    loops=[("i", "N"), ("j", "M"), ("k", "K")],
    body="C[i*M + j] += A[i*K + k] * B[k*M + j];", output="C", reduction={"k"},
)
GEMM_POLY = PolyKernel(
    name="gemm", order=["i", "j", "k"], domain="0<=i<N and 0<=j<M and 0<=k<K",
    writes=[("C", "i,j")], reads=[("A", "i,k"), ("B", "k,j"), ("C", "i,j")],
    params=["N", "M", "K"], sizes={"N": 512, "M": 512, "K": 512},
)

# ---- SYRK : C = A * A^T  (full, rectangular) ------------------------------
SYRK = Kernel(
    name="syrk", sizes={"N": 512, "K": 512},
    arrays={"A": ("N", "K"), "C": ("N", "N")},
    loops=[("i", "N"), ("j", "N"), ("k", "K")],
    body="C[i*N + j] += A[i*K + k] * A[j*K + k];", output="C", reduction={"k"},
)
SYRK_POLY = PolyKernel(
    name="syrk", order=["i", "j", "k"], domain="0<=i<N and 0<=j<N and 0<=k<K",
    writes=[("C", "i,j")], reads=[("A", "i,k"), ("A", "j,k"), ("C", "i,j")],
    params=["N", "K"], sizes={"N": 512, "K": 512},
)

# ---- SYR2K : C = A * B^T + B * A^T  (full) --------------------------------
SYR2K = Kernel(
    name="syr2k", sizes={"N": 512, "K": 512},
    arrays={"A": ("N", "K"), "B": ("N", "K"), "C": ("N", "N")},
    loops=[("i", "N"), ("j", "N"), ("k", "K")],
    body="C[i*N + j] += A[i*K + k] * B[j*K + k] + B[i*K + k] * A[j*K + k];",
    output="C", reduction={"k"},
)
SYR2K_POLY = PolyKernel(
    name="syr2k", order=["i", "j", "k"], domain="0<=i<N and 0<=j<N and 0<=k<K",
    writes=[("C", "i,j")],
    reads=[("A", "i,k"), ("B", "j,k"), ("B", "i,k"), ("A", "j,k"), ("C", "i,j")],
    params=["N", "K"], sizes={"N": 512, "K": 512},
)

# ---- FLOYD-WARSHALL : in-place, k MUST stay outermost ---------------------
# Showcases the legality engine: any reorder moving k inward, or parallel(k),
# is illegal (D[i][j] at step k depends on D[i][k]/D[k][j] from the same k).
FLOYD = Kernel(
    name="floydwarshall", sizes={"N": 512},
    arrays={"D": ("N", "N")},
    loops=[("k", "N"), ("i", "N"), ("j", "N")],
    body="D[i*N + j] = MIN(D[i*N + j], D[i*N + k] + D[k*N + j]);",
    output="D", reduction={"k"}, reset="reinit",
)
FLOYD_POLY = PolyKernel(
    name="floydwarshall", order=["k", "i", "j"],
    domain="0<=k<N and 0<=i<N and 0<=j<N",
    writes=[("D", "i,j")], reads=[("D", "i,j"), ("D", "i,k"), ("D", "k,j")],
    params=["N"], sizes={"N": 512},
)

# ---- real PolyBench triangular variants (j <= i lower triangle) -----------
SYRK_TRI = Kernel(
    name="syrk_tri", sizes={"N": 512, "K": 512},
    arrays={"A": ("N", "K"), "C": ("N", "N")},
    loops=[("i", "N"), ("j", "i+1"), ("k", "K")],
    body="C[i*N + j] += A[i*K + k] * A[j*K + k];", output="C", reduction={"k"},
)
SYRK_TRI_POLY = PolyKernel(
    name="syrk_tri", order=["i", "j", "k"], domain="0<=i<N and 0<=j<=i and 0<=k<K",
    writes=[("C", "i,j")], reads=[("A", "i,k"), ("A", "j,k"), ("C", "i,j")],
    params=["N", "K"], sizes={"N": 512, "K": 512},
)
SYR2K_TRI = Kernel(
    name="syr2k_tri", sizes={"N": 512, "K": 512},
    arrays={"A": ("N", "K"), "B": ("N", "K"), "C": ("N", "N")},
    loops=[("i", "N"), ("j", "i+1"), ("k", "K")],
    body="C[i*N + j] += A[i*K + k] * B[j*K + k] + B[i*K + k] * A[j*K + k];",
    output="C", reduction={"k"},
)
SYR2K_TRI_POLY = PolyKernel(
    name="syr2k_tri", order=["i", "j", "k"], domain="0<=i<N and 0<=j<=i and 0<=k<K",
    writes=[("C", "i,j")],
    reads=[("A", "i,k"), ("B", "j,k"), ("B", "i,k"), ("A", "j,k"), ("C", "i,j")],
    params=["N", "K"], sizes={"N": 512, "K": 512},
)

# ---- MONTE CARLO equity-curve simulation (from ELVIS robustness testing) ---
# M independent simulated equity curves of T trades each. The trade loop is a
# sequential recurrence (equity[s][t] = equity[s][t-1] * (1 + return)); the
# simulation loop is fully independent. So parallel(s) is LEGAL (near-linear
# speedup across cores) and parallel(t) is correctly REJECTED (loop-carried).
# This is the one computation in ELVIS with a real, legal ComPilot win: today it
# runs as a joblib process pool over num_simulations; here it's a single nest the
# agent can tile/parallelize and we measure clang+OpenMP wall-clock against it.
MONTECARLO = Kernel(
    name="montecarlo", sizes={"M": 16384, "T": 1024},
    arrays={"ret": ("M", "T"), "equity": ("M", "T")},
    loops=[("s", "M"), ("t", "1", "T")],
    body="equity[s*T + t] = equity[s*T + (t-1)] * (1.0 + 0.01 * (ret[s*T + t] - 0.5));",
    output="equity", reset="reinit",
)
MONTECARLO_POLY = PolyKernel(
    name="montecarlo", order=["s", "t"], domain="0<=s<M and 1<=t<T",
    writes=[("equity", "s,t")], reads=[("equity", "s,t-1"), ("ret", "s,t")],
    params=["M", "T"], sizes={"M": 16384, "T": 1024},
)

# ---- DISTANCE MATRIX : zvec pairwise squared-Euclidean (alibaba/zvec) -------
# out[i][j] = sum_k (m[i][k] - q[j][k])^2  over M query x N database vectors of
# dim K. This is zvec's euclidean_distance_matrix.h hot loop (the one they
# hand-tune with AVX512). Compute-bound (M*N*K mul-adds), so unlike montecarlo
# it tiles for cache and parallelizes for a real, non-bandwidth-capped win.
# Same dependence shape as SYRK: parallel(i)/parallel(j) legal, parallel(k)
# (the reduction) is correctly rejected.
DISTMATRIX = Kernel(
    name="distmatrix", sizes={"M": 1024, "N": 1024, "K": 128},
    arrays={"m": ("M", "K"), "q": ("N", "K"), "out": ("M", "N")},
    loops=[("i", "M"), ("j", "N"), ("k", "K")],
    body="out[i*N + j] += (m[i*K + k] - q[j*K + k]) * (m[i*K + k] - q[j*K + k]);",
    output="out", reduction={"k"},
)
DISTMATRIX_POLY = PolyKernel(
    name="distmatrix", order=["i", "j", "k"], domain="0<=i<M and 0<=j<N and 0<=k<K",
    writes=[("out", "i,j")], reads=[("m", "i,k"), ("q", "j,k"), ("out", "i,j")],
    params=["M", "N", "K"], sizes={"M": 1024, "N": 1024, "K": 128},
)

REGISTRY = {
    "gemm": (GEMM, GEMM_POLY),
    "distmatrix": (DISTMATRIX, DISTMATRIX_POLY),
    "syrk": (SYRK, SYRK_POLY),
    "syr2k": (SYR2K, SYR2K_POLY),
    "syrk_tri": (SYRK_TRI, SYRK_TRI_POLY),
    "syr2k_tri": (SYR2K_TRI, SYR2K_TRI_POLY),
    "floydwarshall": (FLOYD, FLOYD_POLY),
    "montecarlo": (MONTECARLO, MONTECARLO_POLY),
}
KERNELS = {name: ek for name, (ek, _) in REGISTRY.items()}


# ---- multi-statement kernels (sequence of statements; see multikernel.py) ----
def _twomm():
    from .multikernel import MStmt, MultiKernel
    S = 256
    s0 = MStmt(loops=[("i", "N"), ("j", "M"), ("k", "K")],
               body="tmp[i*M+j] += A[i*K+k]*B[k*M+j];", output="tmp", reduction={"k"},
               poly=PolyKernel("s0", ["i", "j", "k"], "0<=i<N and 0<=j<M and 0<=k<K",
                               [("tmp", "i,j")], [("A", "i,k"), ("B", "k,j"), ("tmp", "i,j")],
                               ["N", "M", "K"]))
    s1 = MStmt(loops=[("i", "N"), ("j", "L"), ("l", "M")],
               body="D[i*L+j] += tmp[i*M+l]*C[l*L+j];", output="D", reduction={"l"},
               poly=PolyKernel("s1", ["i", "j", "l"], "0<=i<N and 0<=j<L and 0<=l<M",
                               [("D", "i,j")], [("tmp", "i,l"), ("C", "l,j"), ("D", "i,j")],
                               ["N", "L", "M"]))
    return MultiKernel("2mm", {"N": S, "M": S, "K": S, "L": S},
                       {"A": ("N", "K"), "B": ("K", "M"), "tmp": ("N", "M"),
                        "C": ("M", "L"), "D": ("N", "L")}, [s0, s1], final="D")


def _poly(name, order, dom, w, r):
    return PolyKernel(name, order, dom, w, r, ["N"])


def _3mm():
    """E = A·B, F = C·D, G = E·F with PolyBench's rectangular NI..NM shapes."""
    from .multikernel import MStmt, MultiKernel
    S = 200

    def mm(out, a, b, I, J, K):                     # out[I][J] += a[I][K] * b[K][J]
        return MStmt(loops=[("i", I), ("j", J), ("k", K)],
                     body=f"{out}[i*{J}+j] += {a}[i*{K}+k]*{b}[k*{J}+j];", output=out, reduction={"k"},
                     poly=PolyKernel(out, ["i", "j", "k"], f"0<=i<{I} and 0<=j<{J} and 0<=k<{K}",
                                     [(out, "i,j")], [(a, "i,k"), (b, "k,j"), (out, "i,j")], [I, J, K]))
    arrays = {"A": ("NI", "NK"), "B": ("NK", "NJ"), "C": ("NJ", "NM"), "D": ("NM", "NL"),
              "E": ("NI", "NJ"), "F": ("NJ", "NL"), "G": ("NI", "NL")}
    return MultiKernel("3mm", {"NI": S, "NJ": S, "NK": S, "NL": S, "NM": S}, arrays,
                       [mm("E", "A", "B", "NI", "NJ", "NK"), mm("F", "C", "D", "NJ", "NL", "NM"),
                        mm("G", "E", "F", "NI", "NL", "NJ")], final="G")


def _matvec_kernel(name, S, arrays, stmts, final):
    from .multikernel import MultiKernel
    return MultiKernel(name, {"N": S}, arrays, stmts, final=final)


def _mvt():
    from .multikernel import MStmt
    S = 2000
    d = "0<=i<N and 0<=j<N"
    arrays = {"A": ("N", "N"), "x1": ("N", 1), "x2": ("N", 1), "y1": ("N", 1), "y2": ("N", 1)}
    s0 = MStmt([("i", "N"), ("j", "N")], "x1[i] += A[i*N+j]*y1[j];", "x1", reduction={"j"},
               poly=_poly("x1", ["i", "j"], d, [("x1", "i")], [("A", "i,j"), ("y1", "j"), ("x1", "i")]))
    s1 = MStmt([("i", "N"), ("j", "N")], "x2[i] += A[j*N+i]*y2[j];", "x2", reduction={"j"},
               poly=_poly("x2", ["i", "j"], d, [("x2", "i")], [("A", "j,i"), ("y2", "j"), ("x2", "i")]))
    return _matvec_kernel("mvt", S, arrays, [s0, s1], "x1")


def _atax():
    """y = Aᵀ(A·x) with A M×N (PolyBench shapes)."""
    from .multikernel import MStmt, MultiKernel
    S = 2000
    d = "0<=i<M and 0<=j<N"
    arrays = {"A": ("M", "N"), "x": ("N", 1), "tmp": ("M", 1), "y": ("N", 1)}
    s0 = MStmt([("i", "M"), ("j", "N")], "tmp[i] += A[i*N+j]*x[j];", "tmp", reduction={"j"},
               poly=PolyKernel("tmp", ["i", "j"], d, [("tmp", "i")],
                               [("A", "i,j"), ("x", "j"), ("tmp", "i")], ["M", "N"]))
    s1 = MStmt([("i", "M"), ("j", "N")], "y[j] += A[i*N+j]*tmp[i];", "y", reduction={"i"},
               poly=PolyKernel("y", ["i", "j"], d, [("y", "j")],
                               [("A", "i,j"), ("tmp", "i"), ("y", "j")], ["M", "N"]))
    return MultiKernel("atax", {"M": S, "N": S}, arrays, [s0, s1], final="y")


def _bicg():
    """s = Aᵀ·r, q = A·p with A N×M (PolyBench shapes)."""
    from .multikernel import MStmt, MultiKernel
    S = 2000
    d = "0<=i<N and 0<=j<M"
    arrays = {"A": ("N", "M"), "r": ("N", 1), "p": ("M", 1), "sv": ("M", 1), "q": ("N", 1)}
    s0 = MStmt([("i", "N"), ("j", "M")], "sv[j] += A[i*M+j]*r[i];", "sv", reduction={"i"},
               poly=PolyKernel("sv", ["i", "j"], d, [("sv", "j")],
                               [("A", "i,j"), ("r", "i"), ("sv", "j")], ["N", "M"]))
    s1 = MStmt([("i", "N"), ("j", "M")], "q[i] += A[i*M+j]*p[j];", "q", reduction={"j"},
               poly=PolyKernel("q", ["i", "j"], d, [("q", "i")],
                               [("A", "i,j"), ("p", "j"), ("q", "i")], ["N", "M"]))
    return MultiKernel("bicg", {"N": S, "M": S}, arrays, [s0, s1], final="q")


def _gesummv():
    from .multikernel import MStmt
    S = 2000
    d = "0<=i<N and 0<=j<N"
    arrays = {"A": ("N", "N"), "B": ("N", "N"), "x": ("N", 1), "tmp": ("N", 1), "y": ("N", 1)}
    s0 = MStmt([("i", "N"), ("j", "N")], "tmp[i] += A[i*N+j]*x[j];", "tmp", reduction={"j"},
               poly=_poly("tmp", ["i", "j"], d, [("tmp", "i")], [("A", "i,j"), ("x", "j"), ("tmp", "i")]))
    s1 = MStmt([("i", "N"), ("j", "N")], "y[i] += B[i*N+j]*x[j];", "y", reduction={"j"},
               poly=_poly("y", ["i", "j"], d, [("y", "i")], [("B", "i,j"), ("x", "j"), ("y", "i")]))
    return _matvec_kernel("gesummv", S, arrays, [s0, s1], "y")


def gesummv_fused():
    """gesummv as one fused nest (both matvecs share `x`) — for the fusion demo."""
    from .multikernel import MStmt, MultiKernel
    S = 2000
    arrays = {"A": ("N", "N"), "B": ("N", "N"), "x": ("N", 1), "tmp": ("N", 1), "y": ("N", 1)}
    f = MStmt([("i", "N"), ("j", "N")],
              "tmp[i] += A[i*N+j]*x[j]; y[i] += B[i*N+j]*x[j];", "tmp", reduction={"j"},
              extra_outputs=["y"],
              poly=_poly("f", ["i", "j"], "0<=i<N and 0<=j<N",
                         [("tmp", "i"), ("y", "i")],
                         [("A", "i,j"), ("B", "i,j"), ("x", "j"), ("tmp", "i"), ("y", "i")]))
    return MultiKernel("gesummv_fused", {"N": S}, arrays, [f], "y")


def _gemver():
    from .multikernel import MStmt, MultiKernel
    S = 1000
    d2 = "0<=i<N and 0<=j<N"
    arrays = {x: ("N", "N") if x == "A" else ("N", 1)
              for x in ("A", "u1", "v1", "u2", "v2", "x", "y", "z", "w")}
    # S1: A = A + u1*v1^T + u2*v2^T  (in-place, element-wise)
    s1 = MStmt([("i", "N"), ("j", "N")], "A[i*N+j] += u1[i]*v1[j] + u2[i]*v2[j];", "A",
               reset="reinit",
               poly=_poly("A", ["i", "j"], d2, [("A", "i,j")],
                          [("u1", "i"), ("v1", "j"), ("u2", "i"), ("v2", "j"), ("A", "i,j")]))
    # S2: x = x + beta * A^T * y   (beta = 1.5)
    s2 = MStmt([("i", "N"), ("j", "N")], "x[i] += 1.5*A[j*N+i]*y[j];", "x", reduction={"j"},
               reset="reinit",
               poly=_poly("x", ["i", "j"], d2, [("x", "i")], [("A", "j,i"), ("y", "j"), ("x", "i")]))
    # S3: x = x + z   (accumulate onto S2's x -> reset none)
    s3 = MStmt([("i", "N")], "x[i] += z[i];", "x", reset="none",
               poly=_poly("x", ["i"], "0<=i<N", [("x", "i")], [("x", "i"), ("z", "i")]))
    # S4: w = w + alpha * A * x   (alpha = 1.5)
    s4 = MStmt([("i", "N"), ("j", "N")], "w[i] += 1.5*A[i*N+j]*x[j];", "w", reduction={"j"},
               reset="reinit",
               poly=_poly("w", ["i", "j"], d2, [("w", "i")], [("A", "i,j"), ("x", "j"), ("w", "i")]))
    return MultiKernel("gemver", {"N": S}, arrays, [s1, s2, s3, s4], "w")


def _covariance():
    """Datamining: mean (reduction) -> center data (in-place) -> covariance (triangular matmul)."""
    from .multikernel import MStmt, MultiKernel
    N, M = 600, 600                       # N samples, M features
    arrays = {"data": ("N", "M"), "mean": ("M", 1), "cov": ("M", "M")}
    s1 = MStmt([("j", "M"), ("i", "N")], "mean[j] += data[i*M+j]/(double)N;", "mean", reduction={"i"},
               poly=PolyKernel("mean", ["j", "i"], "0<=j<M and 0<=i<N",
                               [("mean", "j")], [("data", "i,j"), ("mean", "j")], ["N", "M"]))
    s2 = MStmt([("i", "N"), ("j", "M")], "data[i*M+j] -= mean[j];", "data", reset="reinit",
               poly=PolyKernel("data", ["i", "j"], "0<=i<N and 0<=j<M",
                               [("data", "i,j")], [("data", "i,j"), ("mean", "j")], ["N", "M"]))
    s3 = MStmt([("i", "M"), ("j", "i+1"), ("k", "N")], "cov[i*M+j] += data[k*M+i]*data[k*M+j];", "cov",
               reduction={"k"},
               poly=PolyKernel("cov", ["i", "j", "k"], "0<=i<M and 0<=j<=i and 0<=k<N",
                               [("cov", "i,j")], [("data", "k,i"), ("data", "k,j"), ("cov", "i,j")], ["N", "M"]))
    return MultiKernel("covariance", {"N": N, "M": M}, arrays, [s1, s2, s3], "cov")


def _correlation():
    """Datamining: mean -> variance -> stddev (sqrt+guard) -> normalize (in-place)
    -> correlation (triangular matmul). Same shape as covariance with the extra
    stddev normalization; PolyBench's diagonal corr[i][i]=1 falls out of the
    normalized data so no special-case statement is needed."""
    from .multikernel import MStmt, MultiKernel
    N, M = 600, 600                       # N samples, M features
    arrays = {"data": ("N", "M"), "mean": ("M", 1), "var": ("M", 1),
              "stddev": ("M", 1), "corr": ("M", "M")}
    s1 = MStmt([("j", "M"), ("i", "N")], "mean[j] += data[i*M+j]/(double)N;", "mean", reduction={"i"},
               poly=PolyKernel("mean", ["j", "i"], "0<=j<M and 0<=i<N",
                               [("mean", "j")], [("data", "i,j"), ("mean", "j")], ["N", "M"]))
    s2 = MStmt([("j", "M"), ("i", "N")],
               "var[j] += (data[i*M+j]-mean[j])*(data[i*M+j]-mean[j])/(double)N;", "var", reduction={"i"},
               poly=PolyKernel("var", ["j", "i"], "0<=j<M and 0<=i<N",
                               [("var", "j")], [("data", "i,j"), ("mean", "j"), ("var", "j")], ["N", "M"]))
    s3 = MStmt([("j", "M")], "stddev[j] = sqrt(var[j]); if (stddev[j] <= 1e-9) stddev[j] = 1.0;", "stddev",
               poly=PolyKernel("stddev", ["j"], "0<=j<M", [("stddev", "j")], [("var", "j")], ["M"]))
    s4 = MStmt([("i", "N"), ("j", "M")],
               "data[i*M+j] = (data[i*M+j]-mean[j])/(sqrt((double)N)*stddev[j]);", "data", reset="reinit",
               poly=PolyKernel("data", ["i", "j"], "0<=i<N and 0<=j<M",
                               [("data", "i,j")], [("data", "i,j"), ("mean", "j"), ("stddev", "j")], ["N", "M"]))
    s5 = MStmt([("i", "M"), ("j", "i+1"), ("k", "N")], "corr[i*M+j] += data[k*M+i]*data[k*M+j];", "corr",
               reduction={"k"},
               poly=PolyKernel("corr", ["i", "j", "k"], "0<=i<M and 0<=j<=i and 0<=k<N",
                               [("corr", "i,j")], [("data", "k,i"), ("data", "k,j"), ("corr", "i,j")], ["N", "M"]))
    return MultiKernel("correlation", {"N": N, "M": M}, arrays, [s1, s2, s3, s4, s5], "corr")


MULTI_REGISTRY = {"2mm": _twomm, "3mm": _3mm, "mvt": _mvt, "atax": _atax,
                  "bicg": _bicg, "gesummv": _gesummv, "gemver": _gemver,
                  "covariance": _covariance, "correlation": _correlation}


# ---- stencils: a sequential time loop over scheduled spatial sweeps ---------
def _jacobi1d():
    from .stencil import SStmt, StencilKernel
    N, T, d = 2000000, 100, "1<=i<N-1"
    s0 = SStmt([("i", "1", "N-1")], "B[i] = 0.33333*(A[i-1]+A[i]+A[i+1]);",
               PolyKernel("B", ["i"], d, [("B", "i")], [("A", "i")], ["N"]))
    s1 = SStmt([("i", "1", "N-1")], "A[i] = 0.33333*(B[i-1]+B[i]+B[i+1]);",
               PolyKernel("A", ["i"], d, [("A", "i")], [("B", "i")], ["N"]))
    return StencilKernel("jacobi1d", {"N": N, "TSTEPS": T}, {"A": ("N", 1), "B": ("N", 1)},
                         [s0, s1], reset={"A": "reinit", "B": "reinit"}, final="A")


def _jacobi2d():
    from .stencil import SStmt, StencilKernel
    N, T, d = 1000, 50, "1<=i<N-1 and 1<=j<N-1"
    s0 = SStmt([("i", "1", "N-1"), ("j", "1", "N-1")],
               "B[i*N+j] = 0.2*(A[i*N+j]+A[i*N+j-1]+A[i*N+j+1]+A[(i-1)*N+j]+A[(i+1)*N+j]);",
               PolyKernel("B", ["i", "j"], d, [("B", "i,j")], [("A", "i,j")], ["N"]))
    s1 = SStmt([("i", "1", "N-1"), ("j", "1", "N-1")],
               "A[i*N+j] = 0.2*(B[i*N+j]+B[i*N+j-1]+B[i*N+j+1]+B[(i-1)*N+j]+B[(i+1)*N+j]);",
               PolyKernel("A", ["i", "j"], d, [("A", "i,j")], [("B", "i,j")], ["N"]))
    return StencilKernel("jacobi2d", {"N": N, "TSTEPS": T}, {"A": ("N", "N"), "B": ("N", "N")},
                         [s0, s1], reset={"A": "reinit", "B": "reinit"}, final="A")


def _seidel2d():
    """In-place 9-point Gauss-Seidel: A[i][j] reads already-updated A[i-1]/A[i][j-1],
    so i and j BOTH carry dependences -> naive parallelism is illegal (skewing needed)."""
    from .stencil import SStmt, StencilKernel
    N, T = 1000, 40
    body = ("A[i*N+j] = (A[(i-1)*N+j-1]+A[(i-1)*N+j]+A[(i-1)*N+j+1]"
            "+A[i*N+j-1]+A[i*N+j]+A[i*N+j+1]"
            "+A[(i+1)*N+j-1]+A[(i+1)*N+j]+A[(i+1)*N+j+1])/9.0;")
    s = SStmt([("i", "1", "N-1"), ("j", "1", "N-1")], body,
              PolyKernel("A", ["i", "j"], "1<=i<N-1 and 1<=j<N-1", [("A", "i,j")],
                         [("A", f"{di},{dj}") for di in ("i-1", "i", "i+1") for dj in ("j-1", "j", "j+1")],
                         ["N"]))
    return StencilKernel("seidel2d", {"N": N, "TSTEPS": T}, {"A": ("N", "N")},
                         [s], reset={"A": "reinit"}, final="A")


def _doitgen():
    """3-D: sum[r][q][p] = Σs A[r][q][s]*C4[s][p]; then A = sum (in-place)."""
    from .multikernel import MStmt, MultiKernel
    R = Q = P = 64
    dom = "0<=r<R and 0<=q<Q and 0<=p<P"
    arrays = {"A": ("R", "Q", "P"), "C4": ("P", "P"), "sum": ("R", "Q", "P")}
    s0 = MStmt([("r", "R"), ("q", "Q"), ("p", "P"), ("s", "P")],
               "sum[r*Q*P+q*P+p] += A[r*Q*P+q*P+s]*C4[s*P+p];", "sum", reduction={"s"},
               poly=PolyKernel("sum", ["r", "q", "p", "s"], dom + " and 0<=s<P",
                               [("sum", "r,q,p")], [("A", "r,q,s"), ("C4", "s,p"), ("sum", "r,q,p")],
                               ["R", "Q", "P"]))
    s1 = MStmt([("r", "R"), ("q", "Q"), ("p", "P")], "A[r*Q*P+q*P+p] = sum[r*Q*P+q*P+p];", "A",
               reset="reinit",
               poly=PolyKernel("A", ["r", "q", "p"], dom, [("A", "r,q,p")], [("sum", "r,q,p")],
                               ["R", "Q", "P"]))
    return MultiKernel("doitgen", {"R": R, "Q": Q, "P": P}, arrays, [s0, s1], "A")


MULTI_REGISTRY["doitgen"] = _doitgen

def _heat3d():
    """3-D heat equation: two Jacobi-style sweeps (A->B, B->A) over the N^3 interior.
    Each sweep writes one buffer reading the other, so the spatial loops are fully
    parallel (parallel(i)/tile3d legal) — it's jacobi-2d in one more dimension."""
    from .stencil import SStmt, StencilKernel
    N, T = 64, 40
    d = "1<=i<N-1 and 1<=j<N-1 and 1<=k<N-1"
    loops = [("i", "1", "N-1"), ("j", "1", "N-1"), ("k", "1", "N-1")]

    def sweep(dst, src):
        c = lambda di, dj, dk: (f"{src}[({'i'+di})*N*N+({'j'+dj})*N+({'k'+dk})]")
        body = (f"{dst}[i*N*N+j*N+k] = "
                f"0.125*({c('+1','','')}-2.0*{c('','','')}+{c('-1','','')})"
                f"+0.125*({c('','+1','')}-2.0*{c('','','')}+{c('','-1','')})"
                f"+0.125*({c('','','+1')}-2.0*{c('','','')}+{c('','','-1')})"
                f"+{c('','','')};")
        return SStmt(loops, body,
                     PolyKernel(dst, ["i", "j", "k"], d, [(dst, "i,j,k")], [(src, "i,j,k")], ["N"]))

    return StencilKernel("heat3d", {"N": N, "TSTEPS": T}, {"A": ("N", "N", "N"), "B": ("N", "N", "N")},
                         [sweep("B", "A"), sweep("A", "B")],
                         reset={"A": "reinit", "B": "reinit"}, final="A")


def _fdtd2d():
    """2-D FDTD electromagnetics: per time step a boundary source on ey, then ey/ex
    field updates and the hz update. Every sweep reads OTHER arrays and writes one,
    so all four spatial sweeps are fully parallel (parallel(i)/parallel(j)/tile)."""
    from .stencil import SStmt, StencilKernel
    N, T = 500, 40
    P = ["NX", "NY"]                                     # PolyBench: fields are NX×NY, TMAX steps
    s_b = SStmt([("j", "0", "NY")], "ey[0*NY+j] = (double)t;",
                PolyKernel("ey", ["j"], "0<=j<NY", [("ey", "0,j")], [], P))
    s_ey = SStmt([("i", "1", "NX"), ("j", "0", "NY")],
                 "ey[i*NY+j] -= 0.5*(hz[i*NY+j]-hz[(i-1)*NY+j]);",
                 PolyKernel("ey", ["i", "j"], "1<=i<NX and 0<=j<NY",
                            [("ey", "i,j")], [("hz", "i,j"), ("hz", "i-1,j")], P))
    s_ex = SStmt([("i", "0", "NX"), ("j", "1", "NY")],
                 "ex[i*NY+j] -= 0.5*(hz[i*NY+j]-hz[i*NY+(j-1)]);",
                 PolyKernel("ex", ["i", "j"], "0<=i<NX and 1<=j<NY",
                            [("ex", "i,j")], [("hz", "i,j"), ("hz", "i,j-1")], P))
    s_hz = SStmt([("i", "0", "NX-1"), ("j", "0", "NY-1")],
                 "hz[i*NY+j] -= 0.7*(ex[i*NY+(j+1)]-ex[i*NY+j]+ey[(i+1)*NY+j]-ey[i*NY+j]);",
                 PolyKernel("hz", ["i", "j"], "0<=i<NX-1 and 0<=j<NY-1",
                            [("hz", "i,j")], [("ex", "i,j+1"), ("ex", "i,j"),
                                              ("ey", "i+1,j"), ("ey", "i,j")], P))
    return StencilKernel("fdtd2d", {"NX": N, "NY": N, "TSTEPS": T},
                         {"ex": ("NX", "NY"), "ey": ("NX", "NY"), "hz": ("NX", "NY")},
                         [s_b, s_ey, s_ex, s_hz],
                         reset={"ex": "reinit", "ey": "reinit", "hz": "reinit"}, final="hz")


def _adi():
    """ADI (alternating-direction implicit). Each time step: a column sweep then a
    row sweep, each a forward Thomas elimination + a back substitution. Backward
    sweeps run over an ascending index jj with j=N-2-jj so the forward-only emitter
    stays correct. The sweep direction is carried (parallel on the orthogonal i);
    constants chosen so the a*p+b denominator stays positive (no blow-up)."""
    from .stencil import SStmt, StencilKernel
    N, T = 256, 40
    I = ("i", "1", "N-1")
    # constants: a=1,b=4,c=1,d=1,f=1,(1+2d)=3  -> a*p+b stays in [~3.75,4]
    cs_b = SStmt([I], "v[0*N+i]=1.0; p[i*N+0]=0.0; q[i*N+0]=1.0;",
                 PolyKernel("p", ["i"], "1<=i<N-1", [("v", "0,i"), ("p", "i,0"), ("q", "i,0")], [], ["N"]))
    cs_f = SStmt([I, ("j", "1", "N-1")],
                 "double den=p[i*N+(j-1)]+4.0; p[i*N+j]=-1.0/den;"
                 " q[i*N+j]=(-u[j*N+(i-1)]+3.0*u[j*N+i]-u[j*N+(i+1)]-q[i*N+(j-1)])/den;",
                 PolyKernel("p", ["i", "j"], "1<=i<N-1 and 1<=j<N-1",
                            [("p", "i,j"), ("q", "i,j")], [("p", "i,j-1"), ("q", "i,j-1")], ["N"]))
    cs_eb = SStmt([I], "v[(N-1)*N+i]=1.0;",
                  PolyKernel("v", ["i"], "1<=i<N-1", [("v", "N-1,i")], [], ["N"]))
    cs_b2 = SStmt([I, ("jj", "0", "N-2")],
                  "int j=N-2-jj; v[j*N+i]=p[i*N+j]*v[(j+1)*N+i]+q[i*N+j];",
                  PolyKernel("v", ["i", "jj"], "1<=i<N-1 and 0<=jj<N-2",
                             [("v", "N-2-jj,i")], [("v", "N-1-jj,i")], ["N"]))
    rs_b = SStmt([I], "u[i*N+0]=1.0; p[i*N+0]=0.0; q[i*N+0]=1.0;",
                 PolyKernel("p", ["i"], "1<=i<N-1", [("u", "i,0"), ("p", "i,0"), ("q", "i,0")], [], ["N"]))
    rs_f = SStmt([I, ("j", "1", "N-1")],
                 "double den=p[i*N+(j-1)]+4.0; p[i*N+j]=-1.0/den;"
                 " q[i*N+j]=(-v[(i-1)*N+j]+3.0*v[i*N+j]-v[(i+1)*N+j]-q[i*N+(j-1)])/den;",
                 PolyKernel("p", ["i", "j"], "1<=i<N-1 and 1<=j<N-1",
                            [("p", "i,j"), ("q", "i,j")], [("p", "i,j-1"), ("q", "i,j-1")], ["N"]))
    rs_eb = SStmt([I], "u[i*N+(N-1)]=1.0;",
                  PolyKernel("u", ["i"], "1<=i<N-1", [("u", "i,N-1")], [], ["N"]))
    rs_b2 = SStmt([I, ("jj", "0", "N-2")],
                  "int j=N-2-jj; u[i*N+j]=p[i*N+j]*u[i*N+(j+1)]+q[i*N+j];",
                  PolyKernel("u", ["i", "jj"], "1<=i<N-1 and 0<=jj<N-2",
                             [("u", "i,N-2-jj")], [("u", "i,N-1-jj")], ["N"]))
    return StencilKernel("adi", {"N": N, "TSTEPS": T},
                         {"u": ("N", "N"), "v": ("N", "N"), "p": ("N", "N"), "q": ("N", "N")},
                         [cs_b, cs_f, cs_eb, cs_b2, rs_b, rs_f, rs_eb, rs_b2],
                         reset={"u": "reinit", "v": "reinit"}, final="u")


def _deriche():
    """Deriche recursive Gaussian edge filter (medley). Four IIR passes: horizontal
    L->R and R->L (carried along the row, parallel across rows), a combine, then
    vertical T->B and B->U (carried along the column, parallel across columns) and a
    combine. Backward passes use the ascending-index trick; boundary terms guarded
    by ternaries; stable coefficients (|b1|+|b2|<1). Modeled as a one-step stencil.
    Like PolyBench, the backward passes take two input taps (a3 at +1, a4 at +2)
    and the forward passes one past tap (a2 at -1); the vertical passes reuse the
    horizontal coefficients (PolyBench's a5..a8 are the same values here)."""
    from .stencil import SStmt, StencilKernel
    W, H = 256, 256
    a1, a2, a3, a4, b1, b2, c1, c2 = 0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 1.0, 1.0
    h_lr = SStmt([("i", "0", "W"), ("j", "0", "H")],
                 f"y1[i*H+j] = {a1}*in_[i*H+j] + (j>=1?{a2}*in_[i*H+(j-1)]+{b1}*y1[i*H+(j-1)]:0.0)"
                 f" + (j>=2?{b2}*y1[i*H+(j-2)]:0.0);",
                 PolyKernel("y1", ["i", "j"], "0<=i<W and 0<=j<H",
                            [("y1", "i,j")], [("y1", "i,j-1"), ("y1", "i,j-2")], ["W", "H"]))
    h_rl = SStmt([("i", "0", "W"), ("jj", "0", "H")],
                 f"int j=H-1-jj; y2[i*H+j] = (j<=H-2?{a3}*in_[i*H+(j+1)]+{b1}*y2[i*H+(j+1)]:0.0)"
                 f" + (j<=H-3?{a4}*in_[i*H+(j+2)]+{b2}*y2[i*H+(j+2)]:0.0);",
                 PolyKernel("y2", ["i", "jj"], "0<=i<W and 0<=jj<H",
                            [("y2", "i,H-1-jj")], [("y2", "i,H-jj"), ("y2", "i,H+1-jj")], ["W", "H"]))
    h_c = SStmt([("i", "0", "W"), ("j", "0", "H")],
                f"imgOut[i*H+j] = {c1}*(y1[i*H+j]+y2[i*H+j]);",
                PolyKernel("imgOut", ["i", "j"], "0<=i<W and 0<=j<H",
                           [("imgOut", "i,j")], [("y1", "i,j"), ("y2", "i,j")], ["W", "H"]))
    v_tb = SStmt([("j", "0", "H"), ("i", "0", "W")],
                 f"y1[i*H+j] = {a1}*imgOut[i*H+j] + (i>=1?{a2}*imgOut[(i-1)*H+j]+{b1}*y1[(i-1)*H+j]:0.0)"
                 f" + (i>=2?{b2}*y1[(i-2)*H+j]:0.0);",
                 PolyKernel("y1", ["j", "i"], "0<=j<H and 0<=i<W",
                            [("y1", "i,j")], [("y1", "i-1,j"), ("y1", "i-2,j")], ["W", "H"]))
    v_bu = SStmt([("j", "0", "H"), ("ii", "0", "W")],
                 f"int i=W-1-ii; y2[i*H+j] = (i<=W-2?{a3}*imgOut[(i+1)*H+j]+{b1}*y2[(i+1)*H+j]:0.0)"
                 f" + (i<=W-3?{a4}*imgOut[(i+2)*H+j]+{b2}*y2[(i+2)*H+j]:0.0);",
                 PolyKernel("y2", ["j", "ii"], "0<=j<H and 0<=ii<W",
                            [("y2", "W-1-ii,j")], [("y2", "W-ii,j"), ("y2", "W+1-ii,j")], ["W", "H"]))
    v_c = SStmt([("j", "0", "H"), ("i", "0", "W")],
                f"imgOut[i*H+j] = {c2}*(y1[i*H+j]+y2[i*H+j]);",
                PolyKernel("imgOut", ["j", "i"], "0<=j<H and 0<=i<W",
                           [("imgOut", "i,j")], [("y1", "i,j"), ("y2", "i,j")], ["W", "H"]))
    return StencilKernel("deriche", {"W": W, "H": H, "TSTEPS": 1},
                         {"in_": ("W", "H"), "imgOut": ("W", "H"), "y1": ("W", "H"), "y2": ("W", "H")},
                         [h_lr, h_rl, h_c, v_tb, v_bu, v_c],
                         reset={"imgOut": "zero", "y1": "zero", "y2": "zero"}, final="imgOut")


STENCIL_REGISTRY = {"jacobi1d": _jacobi1d, "jacobi2d": _jacobi2d, "seidel2d": _seidel2d,
                    "heat3d": _heat3d, "fdtd2d": _fdtd2d, "adi": _adi, "deriche": _deriche}


# ---- imperfect-nest kernels (Track C; see imperfect.py) --------------------
def _trisolv():
    """Lower-triangular solve Lx=b. Imperfect: x[i]=b[i], the reduction
    x[i]-=L[i][j]*x[j] (under j<i), x[i]/=L[i][i]. Both loops are carried
    (i by the x[j] recurrence, j by the reduction into x[i]) -> all naive
    parallelism is correctly rejected; the diagonal boost keeps L[i][i] nonzero."""
    from .imperfect import IStmt, ImperfectKernel
    N = 1000
    I = [("i", "0", "N")]
    stmts = [IStmt(I, "x[i] = b[i];"),
             IStmt(I + [("j", "0", "i")], "x[i] -= L[i*N+j]*x[j];"),
             IStmt(I, "x[i] /= L[i*N+i];")]
    return ImperfectKernel("trisolv", {"N": N}, {"L": ("N", "N"), "x": ("N", 1), "b": ("N", 1)},
                           stmts, final="x",
                           reset={"L": "reinit", "x": "zero"},
                           setup="for(int d_=0;d_<N;d_++) L[d_*N+d_]+=N;")


def _lu():
    """Right-looking LU (no pivoting): A[i][k]/=A[k][k] then the rank-1 update
    A[i][j]-=A[i][k]*A[k][j]. k is the sequential carried loop (parallel(k)
    correctly rejected); i and j are independent foralls (parallel(i)/parallel(j)
    proven legal and run correctly — though at PolyBench sizes the parallel form
    doesn't beat serial -O3, so the search keeps identity). Diagonal boost keeps
    pivots nonzero each rep."""
    from .imperfect import IStmt, ImperfectKernel
    N = 256
    KI = [("k", "0", "N"), ("i", "k+1", "N")]
    stmts = [IStmt(KI, "A[i*N+k] /= A[k*N+k];"),
             IStmt(KI + [("j", "k+1", "N")], "A[i*N+j] -= A[i*N+k]*A[k*N+j];")]
    return ImperfectKernel("lu", {"N": N}, {"A": ("N", "N")}, stmts, final="A",
                           reset={"A": "reinit"}, setup="for(int d_=0;d_<N;d_++) A[d_*N+d_]+=N;")


def _cholesky():
    """Cholesky factorization A=LL^T (lower). Under row i: a j-loop holding a
    k-reduction then a divide, a separate diagonal k-reduction, then sqrt — three
    siblings at differing depths. Fully sequential: i carried (reads A[j][k] from
    earlier rows), j carried (column j reads same-row columns k<j), k a reduction
    -> all parallelism rejected. Strong diagonal boost keeps it SPD so sqrt stays
    real (only the lower triangle + diagonal are ever read)."""
    from .imperfect import IStmt, ImperfectKernel
    N = 256
    I = [("i", "0", "N")]
    IJ = I + [("j", "0", "i")]
    stmts = [IStmt(IJ + [("k", "0", "j")], "A[i*N+j] -= A[i*N+k]*A[j*N+k];"),
             IStmt(IJ, "A[i*N+j] /= A[j*N+j];"),
             IStmt(I + [("k", "0", "i")], "A[i*N+i] -= A[i*N+k]*A[i*N+k];"),
             IStmt(I, "A[i*N+i] = sqrt(A[i*N+i]);")]
    return ImperfectKernel("cholesky", {"N": N}, {"A": ("N", "N")}, stmts, final="A",
                           reset={"A": "reinit"}, setup="for(int d_=0;d_<N;d_++) A[d_*N+d_]+=2*N;")


def _ludcmp():
    """LU decomposition (Crout, no pivot) then forward+back substitution Ax=b.
    Under row i: a lower j-loop (j<i) and an upper j-loop (j>=i), each scalar w +
    k-reduction + store; then a forward solve (y) and a DESCENDING back solve (x).
    i is the sequential carried loop -> parallel(i) rejected. Diagonal boost keeps
    pivots nonzero."""
    from .imperfect import IStmt, ImperfectKernel
    N = 200
    I = [("i", "0", "N")]
    JL = I + [("j", "0", "i")]
    JU = I + [("j", "i", "N")]
    stmts = [
        IStmt(JL, "double w = A[i*N+j];"),
        IStmt(JL + [("k", "0", "j")], "w -= A[i*N+k]*A[k*N+j];"),
        IStmt(JL, "A[i*N+j] = w / A[j*N+j];"),
        IStmt(JU, "double w = A[i*N+j];"),
        IStmt(JU + [("k", "0", "i")], "w -= A[i*N+k]*A[k*N+j];"),
        IStmt(JU, "A[i*N+j] = w;"),
        # forward solve Ly=b (unit lower)
        IStmt(I, "double w = b[i];"),
        IStmt(I + [("j", "0", "i")], "w -= A[i*N+j]*y[j];"),
        IStmt(I, "y[i] = w;"),
        # back solve Ux=y (descending i)
        IStmt([("i", "0", "N", "rev")], "double w = y[i];"),
        IStmt([("i", "0", "N", "rev"), ("j", "i+1", "N")], "w -= A[i*N+j]*x[j];"),
        IStmt([("i", "0", "N", "rev")], "x[i] = w / A[i*N+i];"),
    ]
    return ImperfectKernel("ludcmp", {"N": N},
                           {"A": ("N", "N"), "b": ("N", 1), "x": ("N", 1), "y": ("N", 1)},
                           stmts, final="x",
                           reset={"A": "reinit", "x": "zero", "y": "zero"},
                           setup="for(int d_=0;d_<N;d_++) A[d_*N+d_]+=2*N;")


def _durbin():
    """Levinson-Durbin recursion for a Toeplitz system (carried scalars alpha,beta
    and vector y). Under k: scalar updates, a sum-reduction (i<k), an alpha update,
    then z[i] and a SEPARATE y-copy loop (distinct var i2 forces siblings, not
    fusion — the copy must follow all of z). Fully sequential: parallel(k) and the
    reductions are all rejected."""
    from .imperfect import IStmt, ImperfectKernel
    N = 600
    K = [("k", "1", "N")]
    stmts = [
        IStmt(K, "beta = (1.0-alpha*alpha)*beta;"),
        IStmt(K, "sum = 0.0;"),
        IStmt(K + [("i", "0", "k")], "sum += r[k-i-1]*y[i];"),
        IStmt(K, "alpha = -(r[k]+sum)/beta;"),
        IStmt(K + [("i", "0", "k")], "z[i] = y[i] + alpha*y[k-i-1];"),
        IStmt(K + [("i2", "0", "k")], "y[i2] = z[i2];"),
        IStmt(K, "y[k] = alpha;"),
    ]
    return ImperfectKernel("durbin", {"N": N},
                           {"r": ("N", 1), "y": ("N", 1), "z": ("N", 1)},
                           stmts, final="y",
                           reset={"y": "zero", "z": "zero"},
                           setup="double alpha=-r[0], beta=1.0, sum=0.0; y[0]=-r[0];")


def _gramschmidt():
    """Modified Gram-Schmidt QR (A m*n -> Q,R). Under column k: a scalar
    nrm-reduction (i), R[k][k]=sqrt, a Q-normalize (i2), then a j-loop (j>k) with
    R[k][j]=0, an R-reduction (i), and an A-update (i2). k is the sequential
    carried loop (A[i][j] is updated every k) -> parallel(k) rejected."""
    from .imperfect import IStmt, ImperfectKernel
    M, N = 200, 200
    K = [("k", "0", "N")]
    KJ = K + [("j", "k+1", "N")]
    stmts = [
        IStmt(K, "double nrm = 0.0;"),
        IStmt(K + [("i", "0", "M")], "nrm += A[i*N+k]*A[i*N+k];"),
        IStmt(K, "R[k*N+k] = sqrt(nrm);"),
        IStmt(K + [("i2", "0", "M")], "Q[i2*N+k] = A[i2*N+k]/R[k*N+k];"),
        IStmt(KJ, "R[k*N+j] = 0.0;"),
        IStmt(KJ + [("i", "0", "M")], "R[k*N+j] += Q[i*N+k]*A[i*N+j];"),
        IStmt(KJ + [("i2", "0", "M")], "A[i2*N+j] -= Q[i2*N+k]*R[k*N+j];"),
    ]
    # The labelled `i` loop in code is the R-reduction `R[k][j] += Q[i][k]*A[i][j]`
    # (the A-update uses i2). Model R[k,j] as written+read across i so the i loop
    # carries a dependence and parallel(i) is rejected; parallel(j) stays legal
    # (distinct R[k,j]/A[i,j] per j). k stays sequential via the A[i,j] WAW over k.
    return ImperfectKernel("gramschmidt", {"M": M, "N": N},
                           {"A": ("M", "N"), "R": ("N", "N"), "Q": ("M", "N")},
                           stmts, final="Q",
                           reset={"A": "reinit", "R": "zero", "Q": "zero"})


def _trmm():
    """Triangular matrix multiply B := alpha*A^T*B, A lower-triangular (BLAS).
    Under (i,j): a k-reduction (k>i) into B[i][j], then a scale B[i][j]*=alpha.
    i carries an anti-dependence (row i reads B[k][j], k>i, before row k is
    scaled) -> parallel(i) rejected; columns j are independent -> parallel(j)
    legal and runs correctly."""
    from .imperfect import IStmt, ImperfectKernel
    M, N = 256, 256
    IJ = [("i", "0", "M"), ("j", "0", "N")]
    stmts = [IStmt(IJ + [("k", "i+1", "M")], "B[i*N+j] += A[k*M+i]*B[k*N+j];"),
             IStmt(IJ, "B[i*N+j] *= 1.5;")]
    return ImperfectKernel("trmm", {"M": M, "N": N}, {"A": ("M", "M"), "B": ("M", "N")},
                           stmts, final="B", reset={"B": "reinit"})


def _symm():
    """Symmetric matrix multiply C := alpha*A*B + beta*C, A symmetric (BLAS).
    Under (i,j): a scalar temp2 + a k-loop (k<i) that both scatters into C[k][j]
    and accumulates temp2, then the C[i][j] combine. The C[k][j] scatter makes i
    carry an output dependence -> parallel(i) rejected; columns j independent ->
    parallel(j) legal and runs correctly."""
    from .imperfect import IStmt, ImperfectKernel
    M, N = 256, 256
    IJ = [("i", "0", "M"), ("j", "0", "N")]
    stmts = [
        IStmt(IJ, "double temp2 = 0.0;"),
        IStmt(IJ + [("k", "0", "i")], "C[k*N+j] += 1.5*B[i*N+j]*A[i*M+k];"),
        IStmt(IJ + [("k", "0", "i")], "temp2 += B[k*N+j]*A[i*M+k];"),
        IStmt(IJ, "C[i*N+j] = 1.2*C[i*N+j] + 1.5*B[i*N+j]*A[i*M+i] + 1.5*temp2;"),
    ]
    return ImperfectKernel("symm", {"M": M, "N": N},
                           {"A": ("M", "M"), "B": ("M", "N"), "C": ("M", "N")},
                           stmts, final="C", reset={"C": "reinit"})


def _nussinov():
    """Nussinov RNA folding (dynamic programming, medley). i DESCENDING, j>i:
    several depth-2 max-updates (boundaries + paired base) then a depth-3 k-loop
    max. Both i and j are loop-carried (reads table[i+1][j], table[i][j-1]) -> all
    parallelism rejected. Uses MAX; match score baked into the body."""
    from .imperfect import IStmt, ImperfectKernel
    N = 500
    IJ = [("i", "0", "N", "rev"), ("j", "i+1", "N")]
    stmts = [
        IStmt(IJ, "if (j-1>=0) table[i*N+j] = MAX(table[i*N+j], table[i*N+(j-1)]);"),
        IStmt(IJ, "if (i+1<N) table[i*N+j] = MAX(table[i*N+j], table[(i+1)*N+j]);"),
        IStmt(IJ, "if (j-1>=0 && i+1<N) { double m_=(i<j-1)?((seq[i]+seq[j]==3)?1.0:0.0):0.0;"
                  " table[i*N+j] = MAX(table[i*N+j], table[(i+1)*N+(j-1)]+m_); }"),
        IStmt(IJ + [("k", "i+1", "j")], "table[i*N+j] = MAX(table[i*N+j], table[i*N+k]+table[(k+1)*N+j]);"),
    ]
    return ImperfectKernel("nussinov", {"N": N}, {"table": ("N", "N"), "seq": ("N", 1)},
                           stmts, final="table", reset={"table": "zero"},
                           setup="for(int s_=0;s_<N;s_++) seq[s_]=(double)(s_%4);")


IMPERFECT_REGISTRY = {"trisolv": _trisolv, "lu": _lu, "cholesky": _cholesky, "ludcmp": _ludcmp,
                      "durbin": _durbin, "gramschmidt": _gramschmidt, "trmm": _trmm,
                      "symm": _symm, "nussinov": _nussinov}


# ---- PolyBench size classes (the ×5 instance dimension) --------------------
# MINI..EXTRALARGE are PolyBench/C 4.2.1's own per-kernel dataset sizes (copied
# from each kernel's .h). DEFAULT is this repo's built-in size for each kernel —
# what the agent, tests and MCP use unless told otherwise. Legality is size-
# independent (domains are symbolic), so a size class only changes the concrete
# loop bounds the codegen substitutes. Kernels that aren't from PolyBench
# (distmatrix, montecarlo) scale their DEFAULT sizes by SIZE_CLASSES instead.
SIZE_CLASSES = {"MINI": 1 / 16, "SMALL": 1 / 8, "MEDIUM": 1 / 4, "LARGE": 1.0, "EXTRALARGE": 2.0}
DEFAULT_SIZE = "DEFAULT"
SIZE_NAMES = [DEFAULT_SIZE, *SIZE_CLASSES]

POLYBENCH_DATASETS = {   # PolyBench/C 4.2.1 <kernel>.h: class -> parameter -> size
    "2mm": {"MINI": {"NI": 16, "NJ": 18, "NK": 22, "NL": 24}, "SMALL": {"NI": 40, "NJ": 50, "NK": 70, "NL": 80}, "MEDIUM": {"NI": 180, "NJ": 190, "NK": 210, "NL": 220}, "LARGE": {"NI": 800, "NJ": 900, "NK": 1100, "NL": 1200}, "EXTRALARGE": {"NI": 1600, "NJ": 1800, "NK": 2200, "NL": 2400}},
    "3mm": {"MINI": {"NI": 16, "NJ": 18, "NK": 20, "NL": 22, "NM": 24}, "SMALL": {"NI": 40, "NJ": 50, "NK": 60, "NL": 70, "NM": 80}, "MEDIUM": {"NI": 180, "NJ": 190, "NK": 200, "NL": 210, "NM": 220}, "LARGE": {"NI": 800, "NJ": 900, "NK": 1000, "NL": 1100, "NM": 1200}, "EXTRALARGE": {"NI": 1600, "NJ": 1800, "NK": 2000, "NL": 2200, "NM": 2400}},
    "adi": {"MINI": {"TSTEPS": 20, "N": 20}, "SMALL": {"TSTEPS": 40, "N": 60}, "MEDIUM": {"TSTEPS": 100, "N": 200}, "LARGE": {"TSTEPS": 500, "N": 1000}, "EXTRALARGE": {"TSTEPS": 1000, "N": 2000}},
    "atax": {"MINI": {"M": 38, "N": 42}, "SMALL": {"M": 116, "N": 124}, "MEDIUM": {"M": 390, "N": 410}, "LARGE": {"M": 1900, "N": 2100}, "EXTRALARGE": {"M": 1800, "N": 2200}},
    "bicg": {"MINI": {"M": 38, "N": 42}, "SMALL": {"M": 116, "N": 124}, "MEDIUM": {"M": 390, "N": 410}, "LARGE": {"M": 1900, "N": 2100}, "EXTRALARGE": {"M": 1800, "N": 2200}},
    "cholesky": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "correlation": {"MINI": {"M": 28, "N": 32}, "SMALL": {"M": 80, "N": 100}, "MEDIUM": {"M": 240, "N": 260}, "LARGE": {"M": 1200, "N": 1400}, "EXTRALARGE": {"M": 2600, "N": 3000}},
    "covariance": {"MINI": {"M": 28, "N": 32}, "SMALL": {"M": 80, "N": 100}, "MEDIUM": {"M": 240, "N": 260}, "LARGE": {"M": 1200, "N": 1400}, "EXTRALARGE": {"M": 2600, "N": 3000}},
    "deriche": {"MINI": {"W": 64, "H": 64}, "SMALL": {"W": 192, "H": 128}, "MEDIUM": {"W": 720, "H": 480}, "LARGE": {"W": 4096, "H": 2160}, "EXTRALARGE": {"W": 7680, "H": 4320}},
    "doitgen": {"MINI": {"NQ": 8, "NR": 10, "NP": 12}, "SMALL": {"NQ": 20, "NR": 25, "NP": 30}, "MEDIUM": {"NQ": 40, "NR": 50, "NP": 60}, "LARGE": {"NQ": 140, "NR": 150, "NP": 160}, "EXTRALARGE": {"NQ": 220, "NR": 250, "NP": 270}},
    "durbin": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "fdtd-2d": {"MINI": {"TMAX": 20, "NX": 20, "NY": 30}, "SMALL": {"TMAX": 40, "NX": 60, "NY": 80}, "MEDIUM": {"TMAX": 100, "NX": 200, "NY": 240}, "LARGE": {"TMAX": 500, "NX": 1000, "NY": 1200}, "EXTRALARGE": {"TMAX": 1000, "NX": 2000, "NY": 2600}},
    "floyd-warshall": {"MINI": {"N": 60}, "SMALL": {"N": 180}, "MEDIUM": {"N": 500}, "LARGE": {"N": 2800}, "EXTRALARGE": {"N": 5600}},
    "gemm": {"MINI": {"NI": 20, "NJ": 25, "NK": 30}, "SMALL": {"NI": 60, "NJ": 70, "NK": 80}, "MEDIUM": {"NI": 200, "NJ": 220, "NK": 240}, "LARGE": {"NI": 1000, "NJ": 1100, "NK": 1200}, "EXTRALARGE": {"NI": 2000, "NJ": 2300, "NK": 2600}},
    "gemver": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "gesummv": {"MINI": {"N": 30}, "SMALL": {"N": 90}, "MEDIUM": {"N": 250}, "LARGE": {"N": 1300}, "EXTRALARGE": {"N": 2800}},
    "gramschmidt": {"MINI": {"M": 20, "N": 30}, "SMALL": {"M": 60, "N": 80}, "MEDIUM": {"M": 200, "N": 240}, "LARGE": {"M": 1000, "N": 1200}, "EXTRALARGE": {"M": 2000, "N": 2600}},
    "heat-3d": {"MINI": {"TSTEPS": 20, "N": 10}, "SMALL": {"TSTEPS": 40, "N": 20}, "MEDIUM": {"TSTEPS": 100, "N": 40}, "LARGE": {"TSTEPS": 500, "N": 120}, "EXTRALARGE": {"TSTEPS": 1000, "N": 200}},
    "jacobi-1d": {"MINI": {"TSTEPS": 20, "N": 30}, "SMALL": {"TSTEPS": 40, "N": 120}, "MEDIUM": {"TSTEPS": 100, "N": 400}, "LARGE": {"TSTEPS": 500, "N": 2000}, "EXTRALARGE": {"TSTEPS": 1000, "N": 4000}},
    "jacobi-2d": {"MINI": {"TSTEPS": 20, "N": 30}, "SMALL": {"TSTEPS": 40, "N": 90}, "MEDIUM": {"TSTEPS": 100, "N": 250}, "LARGE": {"TSTEPS": 500, "N": 1300}, "EXTRALARGE": {"TSTEPS": 1000, "N": 2800}},
    "lu": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "ludcmp": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "mvt": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "nussinov": {"MINI": {"N": 60}, "SMALL": {"N": 180}, "MEDIUM": {"N": 500}, "LARGE": {"N": 2500}, "EXTRALARGE": {"N": 5500}},
    "seidel-2d": {"MINI": {"TSTEPS": 20, "N": 40}, "SMALL": {"TSTEPS": 40, "N": 120}, "MEDIUM": {"TSTEPS": 100, "N": 400}, "LARGE": {"TSTEPS": 500, "N": 2000}, "EXTRALARGE": {"TSTEPS": 1000, "N": 4000}},
    "symm": {"MINI": {"M": 20, "N": 30}, "SMALL": {"M": 60, "N": 80}, "MEDIUM": {"M": 200, "N": 240}, "LARGE": {"M": 1000, "N": 1200}, "EXTRALARGE": {"M": 2000, "N": 2600}},
    "syr2k": {"MINI": {"M": 20, "N": 30}, "SMALL": {"M": 60, "N": 80}, "MEDIUM": {"M": 200, "N": 240}, "LARGE": {"M": 1000, "N": 1200}, "EXTRALARGE": {"M": 2000, "N": 2600}},
    "syrk": {"MINI": {"M": 20, "N": 30}, "SMALL": {"M": 60, "N": 80}, "MEDIUM": {"M": 200, "N": 240}, "LARGE": {"M": 1000, "N": 1200}, "EXTRALARGE": {"M": 2000, "N": 2600}},
    "trisolv": {"MINI": {"N": 40}, "SMALL": {"N": 120}, "MEDIUM": {"N": 400}, "LARGE": {"N": 2000}, "EXTRALARGE": {"N": 4000}},
    "trmm": {"MINI": {"M": 20, "N": 30}, "SMALL": {"M": 60, "N": 80}, "MEDIUM": {"M": 200, "N": 240}, "LARGE": {"M": 1000, "N": 1200}, "EXTRALARGE": {"M": 2000, "N": 2600}},
}

# repo kernel -> (PolyBench kernel, {repo size parameter: PolyBench parameter})
POLYBENCH = {
    "gemm": ("gemm", {"N": "NI", "M": "NJ", "K": "NK"}),
    "syrk": ("syrk", {"N": "N", "K": "M"}), "syrk_tri": ("syrk", {"N": "N", "K": "M"}),
    "syr2k": ("syr2k", {"N": "N", "K": "M"}), "syr2k_tri": ("syr2k", {"N": "N", "K": "M"}),
    "floydwarshall": ("floyd-warshall", {"N": "N"}),
    "2mm": ("2mm", {"N": "NI", "M": "NJ", "K": "NK", "L": "NL"}),
    "3mm": ("3mm", {p: p for p in ("NI", "NJ", "NK", "NL", "NM")}),
    "mvt": ("mvt", {"N": "N"}), "gesummv": ("gesummv", {"N": "N"}), "gemver": ("gemver", {"N": "N"}),
    "atax": ("atax", {"M": "M", "N": "N"}), "bicg": ("bicg", {"N": "N", "M": "M"}),
    "covariance": ("covariance", {"N": "N", "M": "M"}), "correlation": ("correlation", {"N": "N", "M": "M"}),
    "doitgen": ("doitgen", {"R": "NR", "Q": "NQ", "P": "NP"}),
    "jacobi1d": ("jacobi-1d", {"N": "N", "TSTEPS": "TSTEPS"}),
    "jacobi2d": ("jacobi-2d", {"N": "N", "TSTEPS": "TSTEPS"}),
    "seidel2d": ("seidel-2d", {"N": "N", "TSTEPS": "TSTEPS"}),
    "heat3d": ("heat-3d", {"N": "N", "TSTEPS": "TSTEPS"}),
    "fdtd2d": ("fdtd-2d", {"NX": "NX", "NY": "NY", "TSTEPS": "TMAX"}),
    "adi": ("adi", {"N": "N", "TSTEPS": "TSTEPS"}),
    "deriche": ("deriche", {"W": "W", "H": "H"}),
    "trisolv": ("trisolv", {"N": "N"}), "lu": ("lu", {"N": "N"}), "cholesky": ("cholesky", {"N": "N"}),
    "ludcmp": ("ludcmp", {"N": "N"}), "durbin": ("durbin", {"N": "N"}),
    "gramschmidt": ("gramschmidt", {"M": "M", "N": "N"}), "trmm": ("trmm", {"M": "M", "N": "N"}),
    "symm": ("symm", {"M": "M", "N": "N"}), "nussinov": ("nussinov", {"N": "N"}),
}


def _sizes(name, default, size):
    """Concrete sizes for `name` at `size`, starting from its DEFAULT sizes."""
    if size == DEFAULT_SIZE:
        return dict(default)
    if name in POLYBENCH:
        pb, rename = POLYBENCH[name]
        table = POLYBENCH_DATASETS[pb][size]
        return {**default, **{p: table[q] for p, q in rename.items()}}
    return _scale(default, SIZE_CLASSES[size])


def _scale(sizes, f):
    return {k: max(2, round(v * f)) for k, v in sizes.items()}


def sized_kernel(name, size=DEFAULT_SIZE):
    """Kernel object(s) for `name` at a size class (see SIZE_NAMES).

    Single-statement -> (exec_kernel, poly_kernel); multi/stencil/imperfect -> the
    kernel object. DEFAULT is the registry's own sizes; MINI..EXTRALARGE are
    PolyBench's dataset tables (scale factors for the non-PolyBench extras).
    """
    if size not in SIZE_NAMES:
        raise ValueError(f"unknown size class {size!r}; pick one of {SIZE_NAMES}")
    if name in REGISTRY:
        ek, pk = REGISTRY[name]
        sizes = _sizes(name, ek.sizes, size)
        return replace(ek, sizes=sizes), replace(pk, sizes={k: sizes[k] for k in pk.sizes})
    if name in STENCIL_REGISTRY:
        sk = STENCIL_REGISTRY[name]()
    elif name in MULTI_REGISTRY:
        sk = MULTI_REGISTRY[name]()
    elif name in IMPERFECT_REGISTRY:
        sk = IMPERFECT_REGISTRY[name]()
    else:
        raise KeyError(name)
    sk.sizes = _sizes(name, sk.sizes, size)        # fresh object per call -> safe to mutate
    return sk

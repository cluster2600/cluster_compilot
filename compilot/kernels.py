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
    from .multikernel import MStmt, MultiKernel
    S = 200
    dom = "0<=i<N and 0<=j<N and 0<=k<N"
    def mm(out, a, b):
        return MStmt(loops=[("i", "N"), ("j", "N"), ("k", "N")],
                     body=f"{out}[i*N+j] += {a}[i*N+k]*{b}[k*N+j];", output=out, reduction={"k"},
                     poly=_poly(out, ["i", "j", "k"], dom,
                                [(out, "i,j")], [(a, "i,k"), (b, "k,j"), (out, "i,j")]))
    arrays = {x: ("N", "N") for x in ("A", "B", "C", "D", "E", "F", "G")}
    return MultiKernel("3mm", {"N": S}, arrays,
                       [mm("E", "A", "B"), mm("F", "C", "D"), mm("G", "E", "F")], final="G")


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
    from .multikernel import MStmt
    S = 2000
    d = "0<=i<N and 0<=j<N"
    arrays = {"A": ("N", "N"), "x": ("N", 1), "tmp": ("N", 1), "y": ("N", 1)}
    s0 = MStmt([("i", "N"), ("j", "N")], "tmp[i] += A[i*N+j]*x[j];", "tmp", reduction={"j"},
               poly=_poly("tmp", ["i", "j"], d, [("tmp", "i")], [("A", "i,j"), ("x", "j"), ("tmp", "i")]))
    s1 = MStmt([("i", "N"), ("j", "N")], "y[j] += A[i*N+j]*tmp[i];", "y", reduction={"i"},
               poly=_poly("y", ["i", "j"], d, [("y", "j")], [("A", "i,j"), ("tmp", "i"), ("y", "j")]))
    return _matvec_kernel("atax", S, arrays, [s0, s1], "y")


def _bicg():
    from .multikernel import MStmt
    S = 2000
    d = "0<=i<N and 0<=j<N"
    arrays = {"A": ("N", "N"), "r": ("N", 1), "p": ("N", 1), "sv": ("N", 1), "q": ("N", 1)}
    s0 = MStmt([("i", "N"), ("j", "N")], "sv[j] += A[i*N+j]*r[i];", "sv", reduction={"i"},
               poly=_poly("sv", ["i", "j"], d, [("sv", "j")], [("A", "i,j"), ("r", "i"), ("sv", "j")]))
    s1 = MStmt([("i", "N"), ("j", "N")], "q[i] += A[i*N+j]*p[j];", "q", reduction={"j"},
               poly=_poly("q", ["i", "j"], d, [("q", "i")], [("A", "i,j"), ("p", "j"), ("q", "i")]))
    return _matvec_kernel("bicg", S, arrays, [s0, s1], "q")


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
                         [("A", "i-1,j"), ("A", "i,j-1"), ("A", "i+1,j"), ("A", "i,j+1"), ("A", "i,j")],
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
    s_b = SStmt([("j", "0", "N")], "ey[0*N+j] = (double)t;",
                PolyKernel("ey", ["j"], "0<=j<N", [("ey", "j")], [], ["N"]))
    s_ey = SStmt([("i", "1", "N"), ("j", "0", "N")],
                 "ey[i*N+j] -= 0.5*(hz[i*N+j]-hz[(i-1)*N+j]);",
                 PolyKernel("ey", ["i", "j"], "1<=i<N and 0<=j<N",
                            [("ey", "i,j")], [("hz", "i,j"), ("hz", "i-1,j")], ["N"]))
    s_ex = SStmt([("i", "0", "N"), ("j", "1", "N")],
                 "ex[i*N+j] -= 0.5*(hz[i*N+j]-hz[i*N+(j-1)]);",
                 PolyKernel("ex", ["i", "j"], "0<=i<N and 1<=j<N",
                            [("ex", "i,j")], [("hz", "i,j"), ("hz", "i,j-1")], ["N"]))
    s_hz = SStmt([("i", "0", "N-1"), ("j", "0", "N-1")],
                 "hz[i*N+j] -= 0.7*(ex[i*N+(j+1)]-ex[i*N+j]+ey[(i+1)*N+j]-ey[i*N+j]);",
                 PolyKernel("hz", ["i", "j"], "0<=i<N-1 and 0<=j<N-1",
                            [("hz", "i,j")], [("ex", "i,j+1"), ("ex", "i,j"),
                                              ("ey", "i+1,j"), ("ey", "i,j")], ["N"]))
    return StencilKernel("fdtd2d", {"N": N, "TSTEPS": T},
                         {"ex": ("N", "N"), "ey": ("N", "N"), "hz": ("N", "N")},
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
                 PolyKernel("p", ["i"], "1<=i<N-1", [("p", "i")], [], ["N"]))
    cs_f = SStmt([I, ("j", "1", "N-1")],
                 "double den=p[i*N+(j-1)]+4.0; p[i*N+j]=-1.0/den;"
                 " q[i*N+j]=(-u[j*N+(i-1)]+3.0*u[j*N+i]-u[j*N+(i+1)]-q[i*N+(j-1)])/den;",
                 PolyKernel("p", ["i", "j"], "1<=i<N-1 and 1<=j<N-1",
                            [("p", "i,j")], [("p", "i,j-1")], ["N"]))
    cs_eb = SStmt([I], "v[(N-1)*N+i]=1.0;",
                  PolyKernel("v", ["i"], "1<=i<N-1", [("v", "i")], [], ["N"]))
    cs_b2 = SStmt([I, ("jj", "0", "N-2")],
                  "int j=N-2-jj; v[j*N+i]=p[i*N+j]*v[(j+1)*N+i]+q[i*N+j];",
                  PolyKernel("v", ["i", "jj"], "1<=i<N-1 and 0<=jj<N-2",
                             [("v", "i,jj")], [("v", "i,jj-1")], ["N"]))
    rs_b = SStmt([I], "u[i*N+0]=1.0; p[i*N+0]=0.0; q[i*N+0]=1.0;",
                 PolyKernel("p", ["i"], "1<=i<N-1", [("p", "i")], [], ["N"]))
    rs_f = SStmt([I, ("j", "1", "N-1")],
                 "double den=p[i*N+(j-1)]+4.0; p[i*N+j]=-1.0/den;"
                 " q[i*N+j]=(-v[(i-1)*N+j]+3.0*v[i*N+j]-v[(i+1)*N+j]-q[i*N+(j-1)])/den;",
                 PolyKernel("p", ["i", "j"], "1<=i<N-1 and 1<=j<N-1",
                            [("p", "i,j")], [("p", "i,j-1")], ["N"]))
    rs_eb = SStmt([I], "u[i*N+(N-1)]=1.0;",
                  PolyKernel("u", ["i"], "1<=i<N-1", [("u", "i")], [], ["N"]))
    rs_b2 = SStmt([I, ("jj", "0", "N-2")],
                  "int j=N-2-jj; u[i*N+j]=p[i*N+j]*u[i*N+(j+1)]+q[i*N+j];",
                  PolyKernel("u", ["i", "jj"], "1<=i<N-1 and 0<=jj<N-2",
                             [("u", "i,jj")], [("u", "i,jj-1")], ["N"]))
    return StencilKernel("adi", {"N": N, "TSTEPS": T},
                         {"u": ("N", "N"), "v": ("N", "N"), "p": ("N", "N"), "q": ("N", "N")},
                         [cs_b, cs_f, cs_eb, cs_b2, rs_b, rs_f, rs_eb, rs_b2],
                         reset={"u": "reinit", "v": "reinit"}, final="u")


def _deriche():
    """Deriche recursive Gaussian edge filter (medley). Four IIR passes: horizontal
    L->R and R->L (carried along the row, parallel across rows), a combine, then
    vertical T->B and B->U (carried along the column, parallel across columns) and a
    combine. Backward passes use the ascending-index trick; boundary terms guarded
    by ternaries; stable coefficients (|b1|+|b2|<1). Modeled as a one-step stencil."""
    from .stencil import SStmt, StencilKernel
    W, H = 256, 256
    a1, a2, a3, a4, b1, b2, c1, c2 = 0.1, 0.1, 0.1, 0.1, 0.3, 0.2, 1.0, 1.0
    h_lr = SStmt([("i", "0", "W"), ("j", "0", "H")],
                 f"y1[i*H+j] = {a1}*in_[i*H+j] + (j>=1?{a2}*in_[i*H+(j-1)]+{b1}*y1[i*H+(j-1)]:0.0)"
                 f" + (j>=2?{b2}*y1[i*H+(j-2)]:0.0);",
                 PolyKernel("y1", ["i", "j"], "0<=i<W and 0<=j<H",
                            [("y1", "i,j")], [("y1", "i,j-1")], ["W", "H"]))
    h_rl = SStmt([("i", "0", "W"), ("jj", "0", "H")],
                 f"int j=H-1-jj; y2[i*H+j] = (j<=H-2?{a3}*in_[i*H+(j+1)]+{b1}*y2[i*H+(j+1)]:0.0)"
                 f" + (j<=H-3?{b2}*y2[i*H+(j+2)]:0.0);",
                 PolyKernel("y2", ["i", "jj"], "0<=i<W and 0<=jj<H",
                            [("y2", "i,jj")], [("y2", "i,jj-1")], ["W", "H"]))
    h_c = SStmt([("i", "0", "W"), ("j", "0", "H")],
                f"imgOut[i*H+j] = {c1}*(y1[i*H+j]+y2[i*H+j]);",
                PolyKernel("imgOut", ["i", "j"], "0<=i<W and 0<=j<H",
                           [("imgOut", "i,j")], [("y1", "i,j"), ("y2", "i,j")], ["W", "H"]))
    v_tb = SStmt([("j", "0", "H"), ("i", "0", "W")],
                 f"y1[i*H+j] = {a1}*imgOut[i*H+j] + (i>=1?{a2}*imgOut[(i-1)*H+j]+{b1}*y1[(i-1)*H+j]:0.0)"
                 f" + (i>=2?{b2}*y1[(i-2)*H+j]:0.0);",
                 PolyKernel("y1", ["j", "i"], "0<=j<H and 0<=i<W",
                            [("y1", "i,j")], [("y1", "i-1,j")], ["W", "H"]))
    v_bu = SStmt([("j", "0", "H"), ("ii", "0", "W")],
                 f"int i=W-1-ii; y2[i*H+j] = (i<=W-2?{a3}*imgOut[(i+1)*H+j]+{b1}*y2[(i+1)*H+j]:0.0)"
                 f" + (i<=W-3?{b2}*y2[(i+2)*H+j]:0.0);",
                 PolyKernel("y2", ["j", "ii"], "0<=j<H and 0<=ii<W",
                            [("y2", "ii,j")], [("y2", "ii-1,j")], ["W", "H"]))
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
    poly = PolyKernel("trisolv", ["i", "j"], "0<=i<N and 0<=j<i",
                      [("x", "i")], [("L", "i,j"), ("x", "j"), ("x", "i")], ["N"])
    return ImperfectKernel("trisolv", {"N": N}, {"L": ("N", "N"), "x": ("N", 1), "b": ("N", 1)},
                           stmts, poly, final="x",
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
    poly = PolyKernel("lu", ["k", "i", "j"], "0<=k<N and k<i<N and k<j<N",
                      [("A", "i,j")], [("A", "i,k"), ("A", "k,j"), ("A", "i,j")], ["N"])
    return ImperfectKernel("lu", {"N": N}, {"A": ("N", "N")}, stmts, poly, final="A",
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
    poly = PolyKernel("cholesky", ["i", "j", "k"], "0<=i<N and 0<=j<i and 0<=k<j",
                      [("A", "i,j")], [("A", "i,k"), ("A", "j,k"), ("A", "i,j")], ["N"])
    return ImperfectKernel("cholesky", {"N": N}, {"A": ("N", "N")}, stmts, poly, final="A",
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
    # binding: the update reads A[k][j] (k<i, an earlier ROW) and A[i][k] (k<i, an
    # earlier COLUMN) -> i and j are both carried; the full-width domain is what
    # makes those cross-row/col writes exist so the dependence is seen (a j<i domain
    # hides them and the engine would wrongly call i parallel).
    poly = PolyKernel("ludcmp", ["i", "j", "k"], "0<=i<N and 0<=j<N and 0<=k<i",
                      [("A", "i,j")], [("A", "k,j"), ("A", "i,k"), ("A", "i,j")], ["N"])
    return ImperfectKernel("ludcmp", {"N": N},
                           {"A": ("N", "N"), "b": ("N", 1), "x": ("N", 1), "y": ("N", 1)},
                           stmts, poly, final="x",
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
    # binding: y[i] is rewritten every step k and the update reads y[k-i-1] -> the
    # write y[i] vs read y[k-i-1] couples both steps (carried k) and elements within
    # a step (carried i, when i=k-i'-1). So all naive parallelism is rejected, which
    # is right: durbin is a fully sequential recurrence.
    poly = PolyKernel("durbin", ["k", "i"], "1<=k<N and 0<=i<k",
                      [("y", "i")], [("y", "k-i-1"), ("y", "i")], ["N"])
    return ImperfectKernel("durbin", {"N": N},
                           {"r": ("N", 1), "y": ("N", 1), "z": ("N", 1)},
                           stmts, poly, final="y",
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
    poly = PolyKernel("gramschmidt", ["k", "j", "i"], "0<=k<N and k<j<N and 0<=i<M",
                      [("A", "i,j"), ("R", "k,j")],
                      [("Q", "i,k"), ("R", "k,j"), ("A", "i,j")], ["M", "N"])
    return ImperfectKernel("gramschmidt", {"M": M, "N": N},
                           {"A": ("M", "N"), "R": ("N", "N"), "Q": ("M", "N")},
                           stmts, poly, final="Q",
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
    poly = PolyKernel("trmm", ["i", "j", "k"], "0<=i<M and 0<=j<N and i<k<M",
                      [("B", "i,j")], [("A", "k,i"), ("B", "k,j")], ["M", "N"])
    return ImperfectKernel("trmm", {"M": M, "N": N}, {"A": ("M", "M"), "B": ("M", "N")},
                           stmts, poly, final="B", reset={"B": "reinit"})


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
    # binding statement: the C[k][j] scatter -> output dep carried on i. The fused
    # `temp2 += ...` reduction over k is modeled by a virtual accumulator acc[i,j]
    # (written+read for all k), so the k loop carries a dependence and parallel(k) is
    # correctly rejected. acc is legality-only; codegen emits the IStmt bodies above.
    poly = PolyKernel("symm", ["i", "j", "k"], "0<=i<M and 0<=j<N and 0<=k<i",
                      [("C", "k,j"), ("acc", "i,j")],
                      [("B", "i,j"), ("A", "i,k"), ("C", "k,j"), ("acc", "i,j")], ["M", "N"])
    return ImperfectKernel("symm", {"M": M, "N": N},
                           {"A": ("M", "M"), "B": ("M", "N"), "C": ("M", "N")},
                           stmts, poly, final="C", reset={"C": "reinit"})


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
    poly = PolyKernel("nussinov", ["i", "j", "k"], "0<=i<N and i<j<N and i<k<j",
                      [("table", "i,j")], [("table", "i,k"), ("table", "k,j")], ["N"])
    return ImperfectKernel("nussinov", {"N": N}, {"table": ("N", "N"), "seq": ("N", 1)},
                           stmts, poly, final="table", reset={"table": "zero"},
                           setup="for(int s_=0;s_<N;s_++) seq[s_]=(double)(s_%4);")


IMPERFECT_REGISTRY = {"trisolv": _trisolv, "lu": _lu, "cholesky": _cholesky, "ludcmp": _ludcmp,
                      "durbin": _durbin, "gramschmidt": _gramschmidt, "trmm": _trmm,
                      "symm": _symm, "nussinov": _nussinov}


# ---- PolyBench size classes (the ×5 instance dimension) --------------------
# PolyBench/C ships five standard dataset sizes per kernel. Legality is size-
# independent (domains are symbolic in N/M/K), so a size class only rescales the
# concrete loop bounds the codegen substitutes. The repo's defaults are LARGE.
# ponytail: uniform linear scale of every dim, not PolyBench's exact per-kernel
# NI/NJ/... tables — swap explicit tuples in here if benchmark-faithful sizes
# matter; the ×5 instance structure and the scheduling problem are identical.
SIZE_CLASSES = {"MINI": 1 / 16, "SMALL": 1 / 8, "MEDIUM": 1 / 4, "LARGE": 1.0, "EXTRALARGE": 2.0}


def _scale(sizes, f):
    return {k: max(2, round(v * f)) for k, v in sizes.items()}


def sized_kernel(name, size="LARGE"):
    """Kernel object(s) for `name` with loop bounds rescaled to a PolyBench size class.

    Single-statement -> (exec_kernel, poly_kernel); multi/stencil -> the
    (Multi|Stencil)Kernel. `size` defaults to LARGE (the registry's own sizes),
    so existing callers are unaffected.
    """
    if size not in SIZE_CLASSES:
        raise ValueError(f"unknown size class {size!r}; pick one of {sorted(SIZE_CLASSES)}")
    f = SIZE_CLASSES[size]
    if name in REGISTRY:
        ek, pk = REGISTRY[name]
        return replace(ek, sizes=_scale(ek.sizes, f)), replace(pk, sizes=_scale(pk.sizes, f))
    if name in STENCIL_REGISTRY:
        sk = STENCIL_REGISTRY[name]()
    elif name in MULTI_REGISTRY:
        sk = MULTI_REGISTRY[name]()
    elif name in IMPERFECT_REGISTRY:
        sk = IMPERFECT_REGISTRY[name]()
    else:
        raise KeyError(name)
    sk.sizes = _scale(sk.sizes, f)        # fresh object per call -> safe to mutate
    return sk

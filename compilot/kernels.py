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

STENCIL_REGISTRY = {"jacobi1d": _jacobi1d, "jacobi2d": _jacobi2d, "seidel2d": _seidel2d}


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
    else:
        raise KeyError(name)
    sk.sizes = _scale(sk.sizes, f)        # fresh object per call -> safe to mutate
    return sk

"""Kernels as schedulable affine loop nests.

Each kernel has two paired specs:
  - Kernel      : execution spec (flat arrays, body, sizes) used by codegen
  - PolyKernel  : polyhedral spec (domain, reads, writes, loop order) used by ISL legality

REGISTRY pairs them by name so the Environment can take a single kernel name.
These are single-statement kernels (output zeroed, then an accumulation nest):
GEMM (C=A·B), SYRK (C=A·Aᵀ), SYR2K (C=A·Bᵀ+B·Aᵀ). Multi-statement PolyBench
kernels arrive with the multi-statement polyhedral model.
"""
from dataclasses import dataclass, field
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

REGISTRY = {
    "gemm": (GEMM, GEMM_POLY),
    "syrk": (SYRK, SYRK_POLY),
    "syr2k": (SYR2K, SYR2K_POLY),
    "syrk_tri": (SYRK_TRI, SYRK_TRI_POLY),
    "syr2k_tri": (SYR2K_TRI, SYR2K_TRI_POLY),
    "floydwarshall": (FLOYD, FLOYD_POLY),
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


MULTI_REGISTRY = {"2mm": _twomm, "3mm": _3mm, "mvt": _mvt, "atax": _atax,
                  "bicg": _bicg, "gesummv": _gesummv}

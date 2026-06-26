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

REGISTRY = {
    "gemm": (GEMM, GEMM_POLY),
    "syrk": (SYRK, SYRK_POLY),
    "syr2k": (SYR2K, SYR2K_POLY),
    "floydwarshall": (FLOYD, FLOYD_POLY),
}
KERNELS = {name: ek for name, (ek, _) in REGISTRY.items()}

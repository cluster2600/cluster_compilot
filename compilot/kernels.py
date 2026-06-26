"""PolyBench-style kernels expressed as schedulable affine loop nests.

A kernel is a perfectly-nested loop body over flat (1-D malloc'd) arrays plus
the metadata the codegen/agent needs: loop variables, their bounds, the body
statement, and which loops carry a dependence (informational — legality is
enforced empirically by the correctness check, not by this field).
"""
from dataclasses import dataclass, field


@dataclass
class Kernel:
    name: str
    sizes: dict           # {"N": 1024, ...}  bound name -> int
    arrays: dict          # {"A": ("N", "K"), ...}  name -> (dim, dim) row-major
    loops: list           # [("i", "N"), ("j", "M"), ("k", "K")]  (var, bound)
    body: str             # innermost statement, uses loop vars + flat indexing
    output: str           # name of the array zeroed before the nest
    reduction: set = field(default_factory=set)  # loops carrying a dependence


# C = A * B  (plain matmul). C is zeroed first, so the i/j/k accumulation nest
# is fully permutable + tileable; only parallelizing the reduction loop k is
# illegal — and the checksum gate catches exactly that.
GEMM = Kernel(
    name="gemm",
    sizes={"N": 512, "M": 512, "K": 512},
    arrays={"A": ("N", "K"), "B": ("K", "M"), "C": ("N", "M")},
    loops=[("i", "N"), ("j", "M"), ("k", "K")],
    body="C[i*M + j] += A[i*K + k] * B[k*M + j];",
    output="C",
    reduction={"k"},
)

KERNELS = {k.name: k for k in (GEMM,)}

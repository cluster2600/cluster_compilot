"""Polyhedral legality engine (ISL) — the faithful core of ComPilot.

Tiramisu checks legality with polyhedral dependence analysis; it wraps ISL. We
use ISL directly (islpy), so this is the *same* legality mechanism, not a
substitute. The LLM may propose any schedule; this module proves whether it
preserves every dependence, and whether a given loop level is parallelizable.

Method (textbook polyhedral legality):
  - Each statement has an iteration domain, read/write access relations, and an
    original schedule (loop order).
  - Dependences D = all (source -> sink) iteration pairs that touch the same
    array element with at least one write, ordered by the *original* schedule.
  - A new schedule theta' is LEGAL iff it preserves every dependence:
        for all (i -> i') in D:  theta'(i)  <<_lex  theta'(i')
    i.e. mapping D through theta' must stay strictly lexicographically forward.
  - A schedule dimension p is PARALLEL iff no dependence is *carried* at p
    (equal in all outer dims, strictly increasing at p).
"""
from dataclasses import dataclass, field
import islpy as isl


@dataclass
class PolyKernel:
    """Single-statement polyhedral spec (multi-statement / fusion comes later)."""
    name: str
    order: list                       # loop vars in original nesting order, e.g. ["i","j","k"]
    domain: str                       # isl set body, e.g. "0<=i<N and 0<=j<M and 0<=k<K"
    writes: list                      # [("C", "i,j"), ...]  array <- index tuple
    reads: list                       # [("A", "i,k"), ("B","k,j"), ("C","i,j")]
    params: list = field(default_factory=list)   # ["N","M","K"]
    sizes: dict = field(default_factory=dict)     # concrete sizes for instantiation

    # ---- access relations as ISL union maps -------------------------------
    def _params_str(self):
        return ("[" + ",".join(self.params) + "] -> ") if self.params else ""

    def _S(self):
        return f"S[{','.join(self.order)}]"

    def _dom_set(self):
        p = self._params_str()
        return isl.UnionSet(f"{p}{{ {self._S()} : {self.domain} }}")

    def _access_umap(self, accesses):
        p = self._params_str()
        parts = [f"{self._S()} -> {arr}[{idx}]" for arr, idx in accesses]
        return isl.UnionMap(f"{p}{{ {'; '.join(parts)} }}")

    def writes_umap(self):
        return self._access_umap(self.writes).intersect_domain(self._dom_set())

    def reads_umap(self):
        return self._access_umap(self.reads).intersect_domain(self._dom_set())

    def original_schedule(self):
        # S[i,j,k] -> [i,j,k]
        p = self._params_str()
        outs = ",".join(self.order)
        return isl.UnionMap(f"{p}{{ {self._S()} -> [{outs}] }}").intersect_domain(self._dom_set())


def dependences(k: PolyKernel) -> isl.UnionMap:
    """All memory-based dependences (RAW, WAR, WAW) as source->sink, in original order."""
    W, R = k.writes_umap(), k.reads_umap()
    dom = k.dom = k._dom_set()
    sched = k.original_schedule()

    def same_elem(a, b):
        # { iter_a -> iter_b : a and b touch the same array element }
        return a.apply_range(b.reverse())

    raw = same_elem(W, R)          # write -> read
    war = same_elem(R, W)          # read  -> write
    waw = same_elem(W, W)          # write -> write
    conflict = raw.union(war).union(waw)
    conflict = conflict.intersect_domain(dom).intersect_range(dom)

    # keep only pairs that are strictly ordered in the ORIGINAL schedule, and drop i==i'
    before = sched.lex_lt_union_map(sched)        # { i -> i' : sched(i) <<lex sched(i') }
    return conflict.intersect(before).coalesce()


def _lex_relation(ndim: int, kind: str) -> isl.Map:
    """Build { [a..] -> [b..] : a <<lex b } (kind='lt') or 'ge' over `ndim` dims."""
    a = [f"a{i}" for i in range(ndim)]
    b = [f"b{i}" for i in range(ndim)]
    lt_terms = []
    for i in range(ndim):
        eqs = " and ".join(f"{a[j]}={b[j]}" for j in range(i)) if i else ""
        t = (eqs + " and " if eqs else "") + f"{a[i]}<{b[i]}"
        lt_terms.append("(" + t + ")")
    lt = " or ".join(lt_terms)
    m = isl.Map(f"{{ [{','.join(a)}] -> [{','.join(b)}] : {lt} }}")
    return m if kind == "lt" else m.reverse()  # 'ge'/reverse gives b<<a i.e. violation side


def is_legal(D: isl.UnionMap, theta: isl.UnionMap):
    """Return (legal: bool, violations: UnionMap) for new schedule theta over deps D."""
    if D.is_empty():
        return True, D
    # map both endpoints of every dependence through theta -> { theta(i) -> theta(i') }
    D_time = D.apply_domain(theta).apply_range(theta)
    # collect as a plain Map (single time space); determine its dimensionality
    maps = []
    D_time.foreach_map(maps.append)
    violations = isl.UnionMap("{ }")
    for m in maps:
        ndim = m.dim(isl.dim_type.out)
        reversed_order = _lex_relation(ndim, "ge")     # { t -> t' : t' <<lex t }  (violated)
        bad = m.intersect(reversed_order)
        if not bad.is_empty():
            violations = violations.union(isl.UnionMap.from_map(bad))
    return violations.is_empty(), violations


def is_parallel(D: isl.UnionMap, theta: isl.UnionMap, level: int) -> bool:
    """True iff no dependence is carried exactly at schedule dimension `level`."""
    if D.is_empty():
        return True
    D_time = D.apply_domain(theta).apply_range(theta)
    maps = []
    D_time.foreach_map(maps.append)
    for m in maps:
        ndim = m.dim(isl.dim_type.out)
        if level >= ndim:
            continue
        a = [f"a{i}" for i in range(ndim)]
        b = [f"b{i}" for i in range(ndim)]
        eq_outer = " and ".join(f"{a[j]}={b[j]}" for j in range(level))
        carried = f"{a[level]}<{b[level]}"
        cond = (eq_outer + " and " if eq_outer else "") + carried
        carried_rel = isl.Map(f"{{ [{','.join(a)}] -> [{','.join(b)}] : {cond} }}")
        if not m.intersect(carried_rel).is_empty():
            return False
    return True

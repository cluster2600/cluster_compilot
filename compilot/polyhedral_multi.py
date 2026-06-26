"""Multi-statement polyhedral model (Track B core).

Generalizes the single-statement engine to several statements. Each statement is
an ISL-tagged tuple (S0[i], S1[i,k], ...). Statements are placed in a common time
space with a 2d+1 schedule (interleaved program-order "beta" constants + loop
coordinates), so the SAME dependence + legality machinery (polyhedral.dependences,
is_legal) works across statements — enabling fusion/distribution/shift legality.

A MultiKernel implements the same interface dependences()/is_legal() consume
(_dom_set, writes_umap, reads_umap, original_schedule), so they're reused as-is.
"""
from dataclasses import dataclass, field
import islpy as isl

from .polyhedral import dependences, is_legal


@dataclass
class Statement:
    name: str          # "S0"
    iv: list           # iteration vars, e.g. ["i"]
    domain: str        # isl constraints, e.g. "0<=i<N"
    writes: list       # [(array, idx_tuple)]
    reads: list        # [(array, idx_tuple)]


@dataclass
class MultiKernel:
    name: str
    stmts: list
    params: list = field(default_factory=list)
    # schedule(): name -> list of time-dim expressions (strings over iv + integer betas)
    sched_map: dict = None

    def _p(self):
        return ("[" + ",".join(self.params) + "] -> ") if self.params else ""

    def _dom_set(self):
        parts = [f"{s.name}[{','.join(s.iv)}] : {s.domain}" for s in self.stmts]
        return isl.UnionSet(f"{self._p()}{{ {'; '.join(parts)} }}")

    def _umap(self, pick):
        parts = [f"{s.name}[{','.join(s.iv)}] -> {arr}[{idx}]"
                 for s in self.stmts for arr, idx in pick(s)]
        m = isl.UnionMap(f"{self._p()}{{ {'; '.join(parts)} }}") if parts else isl.UnionMap(f"{self._p()}{{ }}")
        return m.intersect_domain(self._dom_set())

    def writes_umap(self):
        return self._umap(lambda s: s.writes)

    def reads_umap(self):
        return self._umap(lambda s: s.reads)

    def schedule(self, mapping):
        """mapping: stmt name -> list of time expressions (strings). All same length."""
        parts = [f"{s.name}[{','.join(s.iv)}] -> [{','.join(mapping[s.name])}]" for s in self.stmts]
        return isl.UnionMap(f"{self._p()}{{ {'; '.join(parts)} }}").intersect_domain(self._dom_set())

    def original_schedule(self):
        return self.schedule(self.sched_map)


def legal(mk: MultiKernel, new_mapping):
    """Is `new_mapping` (a per-statement schedule) legal given mk's dependences?"""
    D = dependences(mk)
    return is_legal(D, mk.schedule(new_mapping))[0]

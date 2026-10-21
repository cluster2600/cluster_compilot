"""Union-of-statements polyhedral programs (2d+1 schedules): exact multi-statement
legality and ISL codegen with per-loop pragmas.

Where a single-statement kernel has one iteration domain, a UnionProgram has one
per statement, all placed in a common time space by 2d+1 schedules: interleaved
"beta" constants (a statement's position among its siblings) and loop
coordinates, [b0, e0, b1, e1, ..., bn]. Statements whose betas agree up to depth p
share the loop at depth p. Every dependence between any two statement instances
is then checked by the same polyhedral.is_legal / is_parallel, and ISL's AST
generator scans the whole union, so the loops several statements share stay
shared in the generated C.

Users:
  - imperfect.py  — solvers whose statements sit at different loop depths;
  - multikernel.py — fused statement sequences (fuse / shift).
"""
import re
from dataclasses import dataclass, field

import islpy as isl

from .polyhedral import is_parallel


@dataclass
class UStmt:
    name: str                 # ISL tuple and C macro name ("S0", "S1", ...)
    iters: list               # its loop iterators, outermost first
    domain: str               # ISL constraints over iters (+ params); "" = unconstrained
    reads: list               # [(array, "e1,...")]; index "" = a scalar (0-d array)
    writes: list
    body: str                 # C statement(s) over `iters`
    time: list                # 2d+1 schedule [b0, e0, b1, e1, ..., bn] (ISL expressions)
    labels: list              # loop label of each loop dim (len == number of loops)
    decls: list = field(default_factory=list)   # [(ctype, name)] declared at its innermost loop


@dataclass
class UnionProgram:
    stmts: list
    params: list = field(default_factory=list)

    def _p(self):
        return ("[" + ",".join(self.params) + "] -> ") if self.params else ""

    def _tuple(self, s):
        return f"{s.name}[{','.join(s.iters)}]"

    def dom_of(self, members):
        parts = [self._tuple(self.stmts[k]) + (f" : {self.stmts[k].domain}" if self.stmts[k].domain else "")
                 for k in members]
        return isl.UnionSet(f"{self._p()}{{ {'; '.join(parts)} }}")

    def _dom_set(self):
        return self.dom_of(range(len(self.stmts)))

    def _umap(self, pick):
        parts = [f"{self._tuple(s)} -> {a}[{i}]" for s in self.stmts for a, i in pick(s)]
        if not parts:
            return isl.UnionMap(f"{self._p()}{{ }}")
        return isl.UnionMap(f"{self._p()}{{ {'; '.join(parts)} }}").intersect_domain(self._dom_set())

    def writes_umap(self):
        return self._umap(lambda s: s.writes)

    def reads_umap(self):
        return self._umap(lambda s: s.reads)

    def schedule(self, times=None):
        """θ over the whole union; `times` overrides each statement's 2d+1 vector."""
        times = [s.time for s in self.stmts] if times is None else times
        width = max(len(t) for t in times)
        outs = ",".join(f"t{d}" for d in range(width))
        parts = []
        for s, t in zip(self.stmts, times):
            t = list(t) + ["0"] * (width - len(t))
            cons = " and ".join(f"t{d} = {e}" for d, e in enumerate(t))
            parts.append(f"{self._tuple(s)} -> [{outs}] : {cons}")
        return isl.UnionMap(f"{self._p()}{{ {'; '.join(parts)} }}").intersect_domain(self._dom_set())

    def original_schedule(self):
        return self.schedule()


def betas(time, p):
    """b0..bp of a 2d+1 vector: equal for every statement inside one loop at depth p."""
    return tuple(str(b).strip() for b in time[0:2 * p + 1:2])


def loop_members(prog, times, p, key):
    """Indices of the statements inside the depth-p loop identified by betas `key`."""
    return [k for k, (s, t) in enumerate(zip(prog.stmts, times))
            if len(s.labels) > p and betas(t, p) == key]


def loop_is_parallel(prog, D, theta, times, p, key):
    """No dependence between instances inside this one loop is carried by it.
    Restricting D to the loop's own statements matters: is_parallel on the whole
    union would also see other loops that happen to sit at the same depth."""
    members = loop_members(prog, times, p, key)
    if not members:            # a key that matches no loop must never pass as "parallel"
        raise ValueError(f"no statement is inside a depth-{p} loop with betas {key}")
    dom = prog.dom_of(members)
    return is_parallel(D.intersect_domain(dom).intersect_range(dom), theta, 2 * p + 1)


def tree_times(loop_lists):
    """2d+1 vectors for statements nested the way imperfect.py's tree emitter nests
    them: adjacent statements share a loop while their loop tuples agree. A loop
    tuple is (var, lo, hi) or (var, lo, hi, "rev"); a "rev" loop runs descending,
    i.e. its schedule coordinate is -var."""
    times, open_loops, open_beta, children = [], [], [], [0]
    for loops in loop_lists:
        cp = 0
        while cp < len(open_loops) and cp < len(loops) and tuple(open_loops[cp]) == tuple(loops[cp]):
            cp += 1
        del open_loops[cp:], open_beta[cp:], children[cp + 1:]
        for lv in loops[cp:]:
            d = len(open_loops)
            open_beta.append(children[d])
            children[d] += 1
            open_loops.append(lv)
            children.append(0)
        leaf = children[len(loops)]
        children[len(loops)] += 1
        t = []
        for b, lv in zip(open_beta, loops):
            t += [str(b), f"-{lv[0]}" if len(lv) > 3 and lv[3] == "rev" else lv[0]]
        times.append(t + [str(leaf)])
    return times


def _user_names(node):
    t = node.get_type()
    if t == isl.ast_node_type.user:
        return [node.user_get_expr().get_op_arg(0).get_id().get_name()]
    if t == isl.ast_node_type.for_:
        return _user_names(node.for_get_body())
    if t == isl.ast_node_type.if_:
        out = _user_names(node.if_get_then_node())
        return out + (_user_names(node.if_get_else_node()) if node.if_has_else_node() else [])
    if t == isl.ast_node_type.block:
        kids = node.block_get_children()
        return [n for i in range(kids.n_ast_node()) for n in _user_names(kids.get_at(i))]
    if t == isl.ast_node_type.mark:
        return _user_names(node.mark_get_node())
    return []


def to_c(prog, theta, times, context, pragmas=None, indent="    "):
    """C for the union scheduled by θ (`times` = the 2d+1 vectors θ was built from).

    pragmas: {(p, betas): [lines]} placed on the loop at depth p holding those
    statements. Each statement is a macro binding its iterator names, so bodies are
    used verbatim; scalars a statement declares (UStmt.decls) are declared at the
    top of its innermost loop's body (or of the nest), the scope they had in the
    original C — so they stay private to that loop's iterations, including under
    `omp parallel for`."""
    pragmas = pragmas or {}
    by_name = {s.name: (s, t) for s, t in zip(prog.stmts, times)}
    ast = isl.AstBuild.from_context(context).node_from_schedule_map(theta)

    def decls_at(names, depth):
        seen, out = set(), []
        for n in names:
            s, _ = by_name[n]
            if len(s.labels) - 1 == depth:
                for ctype, var in s.decls:
                    if var not in seen:
                        seen.add(var)
                        out.append(f"{ctype} {var};")
        return out

    def emit(node, pad):
        t = node.get_type()
        if t == isl.ast_node_type.for_:
            it = node.for_get_iterator().to_C_str()
            p = (int(it[1:]) - 1) // 2
            names = list(dict.fromkeys(_user_names(node)))
            key = (p, betas(by_name[names[0]][1], p))
            inner = pad + "  "
            body = [inner + d for d in decls_at(names, p)] + emit(node.for_get_body(), inner)
            if node.for_is_degenerate():
                return [pad + "{", f"{inner}const int {it} = {node.for_get_init().to_C_str()};", *body, pad + "}"]
            head = (f"for (int {it} = {node.for_get_init().to_C_str()}; {node.for_get_cond().to_C_str()}; "
                    f"{it} += {node.for_get_inc().to_C_str()}) {{")
            return [pad + pr for pr in pragmas.get(key, ())] + [pad + head, *body, pad + "}"]
        if t == isl.ast_node_type.if_:
            out = [pad + f"if ({node.if_get_cond().to_C_str()}) {{", *emit(node.if_get_then_node(), pad + "  ")]
            if node.if_has_else_node():
                out += [pad + "} else {", *emit(node.if_get_else_node(), pad + "  ")]
            return out + [pad + "}"]
        if t == isl.ast_node_type.block:
            kids = node.block_get_children()
            return [ln for i in range(kids.n_ast_node()) for ln in emit(kids.get_at(i), pad)]
        if t == isl.ast_node_type.mark:
            return emit(node.mark_get_node(), pad)
        if t == isl.ast_node_type.user:
            return [pad + node.user_get_expr().to_C_str() + ";"]
        raise ValueError(f"unexpected AST node type {t}")

    lines = [f"{indent}{d}" for d in decls_at([s.name for s in prog.stmts], -1)] + emit(ast, indent)
    macros, undefs = [], []
    for s in prog.stmts:
        args = [f"a{n}_" for n in range(len(s.iters))]
        binds = " ".join(f"const int {v} = ({a});" for v, a in zip(s.iters, args))
        macros.append(f"#define {s.name}({', '.join(args)}) do {{ {binds} {s.body} }} while (0)")
        undefs.append(f"#undef {s.name}")
    return "\n".join([*macros, "{", *lines, "}", *undefs])


# ---- reading accesses straight from C bodies (imperfect kernels) -------------

_ACC = re.compile(r"\b([A-Za-z_]\w*)\s*\[([^\[\]]*)\]")
_ASSIGN = re.compile(r"\s*([+\-*/]?=)(?!=)")
_DECL = re.compile(r"\b(double|float|int|long)\s+([A-Za-z_]\w*)\s*=")


def _subst(expr, sizes):
    for k in sorted(sizes, key=len, reverse=True):
        expr = re.sub(rf"\b{k}\b", str(sizes[k]), expr)
    return expr


def _assign_at(text, end):
    """The assignment operator right after position `end`, if any."""
    m = _ASSIGN.match(text, end)
    return m.group(1) if m else None


def setup_scalars(setup):
    """Names of the scalars a kernel's setup code declares at function scope."""
    names = []
    for _, rest in re.findall(r"\b(double|float|int|long)\s+([^;()]+);", setup):
        for part in rest.split(","):
            m = re.match(r"\s*([A-Za-z_]\w*)", part)
            if m:
                names.append(m.group(1))
    return names


def body_accesses(body, arrays, sizes, scalar_ref):
    """(reads, writes) of a C body. Array subscripts are flat (A[i*N+j]); sizes are
    substituted so each index is affine. A compound assignment (+=, ...) both reads
    and writes. Accesses under a condition are kept unconditionally — a
    conservative over-approximation, never a missed dependence. scalar_ref(name)
    -> (array, index) maps a scalar to the 0-d or expanded array that models it."""
    reads, writes = [], []
    for m in _ACC.finditer(body):
        name, idx = m.group(1), m.group(2)
        if name not in arrays:
            raise ValueError(f"unknown array {name!r} in {body!r}")
        acc = (name, _subst(idx, sizes))
        op = _assign_at(body, m.end())
        (writes if op else reads).append(acc)
        if op and op != "=":
            reads.append(acc)
    for m in re.finditer(r"\b([A-Za-z_]\w*)\b", body):
        ref = scalar_ref(m.group(1))
        if ref is None or body[m.end():m.end() + 1] == "[":
            continue
        op = _assign_at(body, m.end())
        is_decl = bool(re.search(rf"\b(double|float|int|long)\s+{m.group(1)}\s*$", body[:m.end()]))
        if is_decl or op:
            writes.append(ref)
            if op and op != "=":
                reads.append(ref)
        else:
            reads.append(ref)
    return reads, writes


# ---- fused statement sequences (multikernel / stencil: fuse, shift) ----------

_INT_ALIAS = re.compile(r"\bint\s+([A-Za-z_]\w*)\s*=\s*([^;]+);")


def _inline_int_aliases(body):
    """Substitute body-local index aliases (`int j=N-2-jj;`, the backward sweeps)
    into the rest of the body, for access parsing only."""
    for name, expr in _INT_ALIAS.findall(body):
        body = body.replace(f"int {name}={expr};", "").replace(f"int {name} = {expr};", "")
        body = re.sub(rf"\b{name}\b", f"({expr})", body)
    return body


def body_statement(name, pk, body, arrays, sizes, time, labels):
    """A UStmt whose domain and accesses come from the C (`body`, with sizes
    substituted) rather than a hand-written spec: across fused statements every
    access matters, including reads of arrays the statement itself never writes."""
    flat = _inline_int_aliases(body)
    local = {nm for _, nm in _DECL.findall(flat)}
    iters = list(pk.order)

    def scalar_ref(nm):                        # statement-local scalars: private per instance
        return (f"{nm}__{name}", ",".join(iters)) if nm in local else None
    reads, writes = body_accesses(flat, arrays, sizes, scalar_ref)
    return UStmt(name, iters, _subst(pk.domain, sizes), reads, writes, body, time, labels)


def fused_nest(items, scheds, sizes, arrays, cache, indent="    "):
    """Schedule a sequence of statements as ONE union, so fuse(L) can share loops.

    items: [(PolyKernel, body)] in program order (the spec supplies each statement's
    loop order and domain; accesses are read from the body); scheds: one schedule
    text each.
    Each statement's transforms apply to it as usual (scheduler.schedule_exprs);
    fuse(L) in statement k puts it inside statement k-1's loops down to and
    including L (by position: k's loop at L's depth joins k-1's loop at that
    depth), after k-1's body there; shift(L, c) offsets L to align the two. The
    whole union — every dependence between any two statements — is then checked,
    and ISL generates the fused loops. `cache` (a dict owned by the caller) keeps
    the union dependences across calls. Caller holds ISL_LOCK. Returns
    (nest, None) or (None, failure dict)."""
    from . import schedule as _schedule
    from .polyhedral import dependences, is_legal
    from .scheduler import schedule_exprs
    if "prog" not in cache:                            # concrete sizes: no parameters left
        stmts = []
        for k, (pk, body) in enumerate(items):         # original order: statement k, then its loops
            time = [str(k)]
            for v in pk.order:
                time += [v, "0"]
            stmts.append(body_statement(f"S{k}", pk, body, arrays, sizes, time, list(pk.order)))
        cache["prog"] = prog = UnionProgram(stmts)
        cache["D"] = dependences(prog)
    prog, D = cache["prog"], cache["D"]

    per = []                                            # (betas, exprs, labels, parallel, unroll)
    for k, ((pk, _), text) in enumerate(zip(items, scheds)):
        ops = _schedule.parse(text) if text.strip() else []
        fuse = [args for op, args in ops if op == "fuse"]
        exprs, labels, par, unroll = schedule_exprs(pk, [(op, a) for op, a in ops if op != "fuse"])
        exprs = [_subst(e, sizes) for e in exprs]
        b = [0] * (len(exprs) + 1)
        if fuse:
            if k == 0 or len(fuse) > 1 or len(fuse[0]) != 1:
                raise ValueError("fuse(L): one per statement, never on the first statement")
            if fuse[0][0] not in labels:
                raise ValueError(f"fuse({fuse[0][0]}): no such loop in statement {k} {labels}")
            p, prev = labels.index(fuse[0][0]), per[k - 1][0]
            if len(prev) - 1 <= p:
                raise ValueError(f"fuse({fuse[0][0]}): statement {k - 1} has no loop at depth {p}")
            b[:p + 1] = prev[:p + 1]
            b[p + 1] = prev[p + 1] + 1
        elif k:
            b[0] = per[k - 1][0][0] + 1
        per.append((b, exprs, labels, par, unroll))
    times = []
    for b, exprs, *_ in per:
        t = []
        for bd, e in zip(b, exprs):
            t += [str(bd), e]
        times.append(t + [str(b[len(exprs)])])

    theta = prog.schedule(times)
    if not is_legal(D, theta)[0]:
        return None, {"status": "illegal", "speedup": None,
                      "detail": "the fused schedule violates a dependence between statements"}
    pragmas = {}
    for k, (b, exprs, labels, par, unroll) in enumerate(per):
        for lbl, q in par:
            key = (q, betas(times[k], q))
            if not loop_is_parallel(prog, D, theta, times, q, key[1]):
                return None, {"status": "parallel_illegal", "stmt": k, "speedup": None,
                              "detail": f"loop {lbl} of statement {k} carries a dependence"}
            lines = pragmas.setdefault(key, [])
            if "#pragma omp parallel for" not in lines:
                lines.insert(0, "#pragma omp parallel for")
        for lbl, q, f in unroll:
            lines = pragmas.setdefault((q, betas(times[k], q)), [])
            line = f"#pragma clang loop unroll_count({f})"
            if any(x.startswith("#pragma clang loop unroll") and x != line for x in lines):
                raise ValueError(f"conflicting unroll factors on the fused loop {lbl}")
            if line not in lines:
                lines.append(line)
    return to_c(prog, theta, times, isl.Set("{ : }"), pragmas, indent), None


def wants_fusion(scheds):
    """True if any statement's schedule uses fuse() (switching to the union path)."""
    return any(re.search(r"^\s*fuse\s*\(", t, re.M) for t in scheds if t)

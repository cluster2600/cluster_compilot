"""Translate a parsed schedule (list of transforms) into an ISL schedule map θ'.

θ' maps each statement iteration to its new logical time vector. The legality
engine (polyhedral.is_legal / is_parallel) proves whether θ' preserves all
dependences, and codegen.isl_nest generates the executed loops FROM THE SAME θ',
so what is proven legal is exactly what runs.

Transforms (per statement): interchange/reorder, reverse, skew, shift, tile/
tile2d/tile3d, plus parallel/unroll (recorded per schedule dimension, no
reordering). Tiling follows Tiramisu: tile2d(i, j, Ti, Tj) yields i_t, j_t, i, j —
the tile loops are inserted together, in argument order, just outside the
outermost of the tiled loops. shift(L, c) offsets loop L's schedule value by c (a
no-op on its own; it aligns statements for fusion). fuse is a multi-statement op
handled by union.fused_nest.

Every loop label an op names is validated here, so a typo in an LLM proposal is
reported as an invalid schedule instead of surfacing later in codegen.
"""
import islpy as isl


def _idx(dims, label):
    for i, d in enumerate(dims):
        if d["label"] == label:
            return i
    raise ValueError(f"loop/label {label!r} not in schedule {[d['label'] for d in dims]}")


def _point(dims, label, op):
    """Index of a point (non-tile) loop; tile loops can't be re-tiled or skewed."""
    i = _idx(dims, label)
    if dims[i]["tile"] is not None:
        raise ValueError(f"{op} applies to point loops, not tile loop {label!r}")
    return i


def _int(s, what):
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{what} must be an integer, got {s!r}") from None


def build_theta(kernel, ops):
    """Return (theta, labels, parallel_levels, unroll_levels) — schedule_exprs as a map."""
    exprs, labels, parallel_levels, unroll_levels = schedule_exprs(kernel, ops)
    params = ("[" + ",".join(kernel.params) + "] -> ") if kernel.params else ""
    outs = ",".join(f"o{i}" for i in range(len(exprs)))
    cons = " and ".join(f"o{i} = {e}" for i, e in enumerate(exprs))
    body = f"S[{','.join(kernel.order)}] -> [{outs}] : {cons}"
    theta = isl.UnionMap(f"{params}{{ {body} }}").intersect_domain(kernel._dom_set())
    return theta, labels, parallel_levels, unroll_levels


def schedule_exprs(kernel, ops):
    """Return (exprs, labels, parallel_levels, unroll_levels).

    exprs: one ISL expression over the original iterators per schedule dimension.

    theta: UnionMap S[iters] -> [o0..on]; labels: loop label per schedule dim;
    parallel_levels: [(label, level)]; unroll_levels: [(label, level, factor)].

    A point dim's value is an affine `expr` over the original iterators. A tile
    dim's value is sign*floor(base/T) + off, where base is the tiled loop's expr
    at tiling time (reverse flips sign/off, so tile loops can be reversed too).
    """
    dims = [{"label": v, "expr": v, "tile": None} for v in kernel.order]
    parallel, unroll = [], []

    def tile_group(pairs):
        pos = [_point(dims, v, "tile") for v, _ in pairs]
        if len(set(pos)) != len(pos):
            raise ValueError(f"tile loops must be distinct, got {[v for v, _ in pairs]}")
        new = []
        for (v, T), i in zip(pairs, pos):
            t = _int(T, "tile size")
            if t <= 0:
                raise ValueError(f"tile size must be positive, got {t}")
            if any(d["label"] == f"{v}_t" for d in dims):
                raise ValueError(f"loop {v!r} is already tiled")
            new.append({"label": f"{v}_t", "tile": (dims[i]["expr"], t), "sign": 1, "off": "0"})
        at = min(pos)
        dims[at:at] = new

    for op, args in ops:
        if op == "reorder":
            if sorted(args) != sorted(d["label"] for d in dims):
                raise ValueError(f"reorder{tuple(args)} is not a permutation of {[d['label'] for d in dims]}")
            dims.sort(key=lambda d: args.index(d["label"]))
        elif op == "interchange":
            ia, ib = _idx(dims, args[0]), _idx(dims, args[1])
            dims[ia], dims[ib] = dims[ib], dims[ia]
        elif op == "reverse":
            d = dims[_idx(dims, args[0])]
            if d["tile"] is None:
                d["expr"] = f"-({d['expr']})"
            else:
                d["sign"], d["off"] = -d["sign"], f"-({d['off']})"
        elif op == "shift":
            if len(args) != 2:
                raise ValueError(f"shift takes (loop, offset), got {args!r}")
            d = dims[_point(dims, args[0], "shift")]
            d["expr"] = f"({d['expr']}) + {_int(args[1], 'shift offset')}"
        elif op == "skew":
            t, s, f = args
            ti, si = _point(dims, t, "skew"), _point(dims, s, "skew")
            if ti == si:
                raise ValueError("skew needs two different loops")
            dims[ti]["expr"] = f"({dims[ti]['expr']}) + {_int(f, 'skew factor')}*({dims[si]['expr']})"
        elif op == "tile":
            tile_group([(args[0], args[1])])
        elif op == "tile2d":
            tile_group([(args[0], args[2]), (args[1], args[3])])
        elif op == "tile3d":
            tile_group([(args[0], args[3]), (args[1], args[4]), (args[2], args[5])])
        elif op == "parallel":
            parallel.append(args[0])
        elif op == "unroll":
            unroll.append((args[0], args[1]))
        else:
            raise ValueError(f"unsupported transform for legality: {op}")

    labels = [d["label"] for d in dims]
    exprs = []
    for d in dims:
        if d["tile"] is None:
            exprs.append(d["expr"])
        else:
            base, T = d["tile"]
            sign = "" if d["sign"] > 0 else "-"
            exprs.append(f"{sign}floor(({base})/{T}) + ({d['off']})")
    parallel_levels = [(lbl, _idx(dims, lbl)) for lbl in parallel]
    unroll_levels = [(lbl, _idx(dims, lbl), _int(f, "unroll factor")) for lbl, f in unroll]
    return exprs, labels, parallel_levels, unroll_levels

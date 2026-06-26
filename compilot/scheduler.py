"""Translate a parsed schedule (list of transforms) into an ISL schedule map θ'.

θ' maps each statement iteration to its new logical time vector. The legality
engine (polyhedral.is_legal / is_parallel) then proves whether θ' preserves all
dependences. Transforms supported here (single-statement): interchange/reorder,
reverse, skew, tile/tile2d/tile3d, plus parallel/unroll (recorded, no reorder).
Fusion/shifting are multi-statement and handled once multi-statement kernels land.
"""
import islpy as isl


def _idx(dims, label):
    for i, d in enumerate(dims):
        if d["label"] == label:
            return i
    raise ValueError(f"loop/label {label!r} not in schedule {[d['label'] for d in dims]}")


def build_theta(kernel, ops):
    """Return (theta: UnionMap, labels: list[str], parallel: list[(label,level)], unroll: list)."""
    dims = [{"label": v, "expr": v, "tile": None} for v in kernel.order]
    parallel, unroll = [], []

    def expand_tiles(pairs):
        for var, T in pairs:
            i = _idx(dims, var)
            outer = {"label": f"{var}_t", "expr": None, "tile": (var, int(T))}
            dims[i:i + 1] = [outer, dims[i]]            # tile loop just outside point loop

    for op, args in ops:
        if op == "reorder":
            dims.sort(key=lambda d: args.index(d["label"]))
        elif op == "interchange":
            ia, ib = _idx(dims, args[0]), _idx(dims, args[1])
            dims[ia], dims[ib] = dims[ib], dims[ia]
        elif op == "reverse":
            dims[_idx(dims, args[0])]["expr"] = f"(-{args[0]})"
        elif op == "skew":
            t, s, f = args
            dims[_idx(dims, t)]["expr"] = f"({t} + {f}*{s})"
        elif op == "tile":
            expand_tiles([(args[0], args[1])])
        elif op == "tile2d":
            expand_tiles([(args[0], args[2]), (args[1], args[3])])
        elif op == "tile3d":
            expand_tiles([(args[0], args[3]), (args[1], args[4]), (args[2], args[5])])
        elif op == "parallel":
            parallel.append(args[0])
        elif op == "unroll":
            unroll.append((args[0], args[1]))
        else:
            raise ValueError(f"unsupported transform for legality: {op}")

    labels = [d["label"] for d in dims]
    params = ("[" + ",".join(kernel.params) + "] -> ") if kernel.params else ""
    outs = ",".join(f"o{i}" for i in range(len(dims)))
    cons = []
    for i, d in enumerate(dims):
        if d["tile"] is None:
            cons.append(f"o{i} = {d['expr']}")
        else:
            var, T = d["tile"]
            cons.append(f"{T}*o{i} <= {var} <= {T}*o{i} + {T - 1}")
    body = f"S[{','.join(kernel.order)}] -> [{outs}] : " + " and ".join(cons)
    theta = isl.UnionMap(f"{params}{{ {body} }}").intersect_domain(kernel._dom_set())
    parallel_levels = [(lbl, labels.index(lbl)) for lbl in parallel]
    return theta, labels, parallel_levels, unroll

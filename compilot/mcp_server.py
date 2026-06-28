"""MCP server for cluster_compilot — exposes the optimizer to Claude Code & Codex.

A minimal stdio JSON-RPC 2.0 server (newline-delimited messages), hand-rolled in
stdlib to match the repo's dependency-light style (no FastMCP/pydantic). Three tools:

  list_kernels()                       -> the schedulable kernels, by category
  check_legality(kernel, schedule)     -> ISL verdict + measured speedup (single-stmt, fast)
  optimize(kernel, backend=mock, ...)  -> run the agent loop, return best speedup + schedule

Run:  python -m compilot.mcp_server      (or the `compilot-mcp` console script)
Wire into Claude Code / Codex: see npm/README.md.

# ponytail: hand-rolled MCP (initialize/tools.list/tools.call only); swap in the
# `mcp` SDK only if we need resources/prompts/streaming.
"""
import json
import sys

SERVER_INFO = {"name": "compilot", "version": "0.1.0"}
DEFAULT_PROTOCOL = "2025-06-18"

TOOLS = [
    {
        "name": "list_kernels",
        "description": "List the schedulable loop-nest kernels by category "
                       "(single-statement, multi-statement, stencil).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_legality",
        "description": "Prove a hand-written schedule's polyhedral legality (ISL) and, if "
                       "legal, compile + run it for a real wall-clock speedup. Single-statement "
                       "kernels only (gemm, syrk, ...); sub-second, no LLM. schedule is the "
                       "transform DSL, e.g. 'reorder(i, k, j)\\ntile2d(i, j, 32, 32)\\nparallel(i_t)'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kernel": {"type": "string", "description": "single-statement kernel name"},
                "schedule": {"type": "string", "description": "transform DSL, one transform per line"},
                "size": {"type": "string",
                         "enum": ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"],
                         "default": "LARGE", "description": "PolyBench dataset size class"},
            },
            "required": ["kernel", "schedule"],
        },
    },
    {
        "name": "optimize",
        "description": "Run the ComPilot agent loop on a kernel and return the best measured "
                       "speedup + winning schedule. backend 'mock' is offline/deterministic and "
                       "needs no API key (default); 'gemini'/'local' drive real LLMs (keys/server "
                       "required) and take minutes. Set moa='gemini:gemini-2.5-flash,local:qwen2.5-coder:32b' "
                       "for Mixture-of-Agents.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kernel": {"type": "string"},
                "backend": {"type": "string", "enum": ["mock", "gemini", "local"], "default": "mock"},
                "model": {"type": "string", "default": "gemini-2.5-flash"},
                "iters": {"type": "integer", "default": 8},
                "candidates": {"type": "integer", "default": 3,
                               "description": "schedules proposed + measured in parallel per turn"},
                "moa": {"type": "string", "default": "",
                        "description": "comma-separated MoA reference specs; empty = single model"},
                "aggregator": {"type": "string", "default": ""},
                "base_url": {"type": "string", "default": "http://localhost:11434/v1"},
                "size": {"type": "string",
                         "enum": ["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"],
                         "default": "LARGE", "description": "PolyBench dataset size class (×5 instances)"},
            },
            "required": ["kernel"],
        },
    },
]


# --- tool implementations (imports are lazy so tools/list never loads islpy) ---

def _make_client(spec, base_url, temperature=0.7):
    """spec is 'backend:model' (split on first ':' so Ollama tags survive)."""
    from .llm import GeminiClient, OpenAIClient, MockClient
    backend, _, model = spec.partition(":")
    if backend == "mock":
        return MockClient()
    if backend == "local":
        return OpenAIClient(model=model, base_url=base_url, temperature=temperature)
    if backend == "gemini":
        return GeminiClient(model=model, temperature=temperature)
    raise ValueError(f"unknown backend in {spec!r} (use gemini:…, local:…, or mock)")


def _list_kernels():
    from .kernels import REGISTRY, MULTI_REGISTRY, STENCIL_REGISTRY
    return {"single": sorted(REGISTRY),
            "multi": sorted(MULTI_REGISTRY),
            "stencil": sorted(STENCIL_REGISTRY)}


def _check_legality(kernel, schedule, size="LARGE"):
    from .backend_isl import environment
    from .kernels import REGISTRY
    if kernel not in REGISTRY:
        return {"error": f"{kernel!r} is multi-statement/stencil or unknown. check_legality "
                         f"supports single-statement kernels {sorted(REGISTRY)}; use optimize "
                         f"for the rest."}
    r = environment(kernel, size).evaluate(schedule)
    return {"status": r.status, "speedup": r.speedup, "detail": r.detail}


def _optimize(kernel, backend="mock", model="gemini-2.5-flash", iters=8, candidates=3,
              moa="", aggregator="", base_url="http://localhost:11434/v1", size="LARGE"):
    from .agent import (run_dialogue, run_dialogue_multi, run_dialogue_moa,
                        run_dialogue_moa_multi)
    from .backend_isl import environment
    from .kernels import MULTI_REGISTRY, STENCIL_REGISTRY, sized_kernel
    from .multikernel import MultiEnvironment
    from .stencil import StencilEnvironment

    base_spec = "mock" if backend == "mock" else f"{backend}:{model}"
    refs = [_make_client(s.strip(), base_url, 0.9) for s in moa.split(",") if s.strip()] if moa else []

    if kernel in MULTI_REGISTRY or kernel in STENCIL_REGISTRY:
        menv = (StencilEnvironment(sized_kernel(kernel, size)) if kernel in STENCIL_REGISTRY
                else MultiEnvironment(sized_kernel(kernel, size)))
        if moa:
            agg = _make_client(aggregator or base_spec, base_url, 0.4)
            sp, best = run_dialogue_moa_multi(menv, refs, agg, max_iters=iters, verbose=False)
        else:
            sp, best = run_dialogue_multi(menv, _make_client(base_spec, base_url), max_iters=iters,
                                          verbose=False)
        sched = "\n".join(f"[stmt {i}] {s.strip() or '(identity)'}" for i, s in enumerate(best or []))
    else:
        env = environment(kernel, size)
        if moa:
            agg = _make_client(aggregator or base_spec, base_url, 0.4)
            sp, sched, _ = run_dialogue_moa(env, refs, agg, max_iters=iters, verbose=False,
                                            candidates_per_turn=candidates)
        else:
            sp, sched, _ = run_dialogue(env, _make_client(base_spec, base_url), max_iters=iters,
                                        verbose=False, candidates_per_turn=candidates)
    return {"best_speedup": round(sp, 3), "schedule": sched.strip()}


def _call_tool(name, args):
    if name == "list_kernels":
        return _list_kernels()
    if name == "check_legality":
        return _check_legality(args["kernel"], args["schedule"], args.get("size", "LARGE"))
    if name == "optimize":
        return _optimize(
            args["kernel"], args.get("backend", "mock"), args.get("model", "gemini-2.5-flash"),
            int(args.get("iters", 8)), int(args.get("candidates", 3)),
            args.get("moa", ""), args.get("aggregator", ""),
            args.get("base_url", "http://localhost:11434/v1"), args.get("size", "LARGE"))
    raise ValueError(f"unknown tool {name!r}")


# --- JSON-RPC / MCP framing ---

def _text_result(obj):
    text = obj if isinstance(obj, str) else json.dumps(obj, indent=2)
    return {"content": [{"type": "text", "text": text}]}


def handle(msg):
    """Map one JSON-RPC message to a response dict, or None for notifications."""
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        proto = (msg.get("params") or {}).get("protocolVersion", DEFAULT_PROTOCOL)
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": proto,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params") or {}
        try:
            return {"jsonrpc": "2.0", "id": mid,
                    "result": _text_result(_call_tool(params["name"], params.get("arguments") or {}))}
        except Exception as e:                      # surface tool errors via isError, not a crash
            return {"jsonrpc": "2.0", "id": mid,
                    "result": {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True}}
    if mid is None:                                 # a notification (e.g. notifications/initialized)
        return None
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()

# @cluster2600/compilot-mcp

MCP server for [cluster_compilot](https://github.com/cluster2600/cluster_compilot) —
an LLM agent proposes loop transformations, a polyhedral engine (ISL) proves
legality, and clang `-O3 +OpenMP` compiles & **times** the result for a real
speedup. This npm package is a thin launcher around the Python server; the heavy
deps (`islpy`, `clang`) live in the Python environment.

## Prerequisites

The Python package must be importable. Clone and install it once:

```bash
git clone https://github.com/cluster2600/cluster_compilot.git
cd cluster_compilot
python3.14 -m venv .venv && . .venv/bin/activate
pip install -e .
brew install libomp        # OpenMP for clang (macOS)
```

Then point the launcher at that interpreter via `COMPILOT_PYTHON` (or run it from
the checkout, where it finds `./.venv/bin/python` automatically).

## Tools

| Tool | What it does |
|---|---|
| `list_kernels` | the schedulable kernels, by category (single / multi / stencil) |
| `check_legality(kernel, schedule)` | prove a hand-written schedule legal (ISL) + compile & time it — single-statement, sub-second, no LLM |
| `optimize(kernel, backend=mock, iters, candidates, moa, aggregator)` | run the agent loop, return best measured speedup + schedule. `mock` is offline/keyless; `gemini`/`local` drive real models |

## Claude Code

```bash
COMPILOT_PYTHON=/path/to/cluster_compilot/.venv/bin/python \
  claude mcp add compilot -- npx -y @cluster2600/compilot-mcp
```

## Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.compilot]
command = "npx"
args = ["-y", "@cluster2600/compilot-mcp"]
env = { COMPILOT_PYTHON = "/path/to/cluster_compilot/.venv/bin/python" }
```

## Try it

> "List the compilot kernels, then check the legality of `reorder(i, k, j)` on gemm."

`check_legality` returns the ISL verdict and the measured speedup; `optimize`
(default `mock` backend) runs the full search offline.

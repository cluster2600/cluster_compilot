# @cluster2600/compilot-mcp

MCP server for [cluster_compilot](https://github.com/cluster2600/cluster_compilot) —
an LLM agent proposes loop transformations, a polyhedral engine (ISL) proves
legality, and clang `-O3 +OpenMP` compiles & **times** the result for a real
speedup. This npm package is a thin launcher around the Python server; the heavy
deps (`islpy`, `clang`) live in the Python environment.

## Prerequisites

The server is the Python package `compilot` (Python 3.14+), and it compiles the
kernels it times with clang + OpenMP. Install both once:

```bash
python3 -m venv ~/.venvs/compilot && ~/.venvs/compilot/bin/pip install compilot
brew install libomp                      # macOS: OpenMP for clang
# sudo apt install clang libomp-dev      # Debian / Ubuntu
```

(Working from a clone of the repo instead? `pip install -e .` in it does the same.)

Then point the launcher at that interpreter via `COMPILOT_PYTHON` (otherwise it
uses `python3` from `PATH`). It deliberately never picks up a `./.venv` or a
`./compilot/` from the directory it is started in — MCP clients launch it inside
whatever project is open, and that project must not be able to swap in its own code.

## Tools

| Tool | What it does |
|---|---|
| `list_kernels` | the schedulable kernels, by category (single / multi / stencil) |
| `check_legality(kernel, schedule)` | prove a hand-written schedule legal (ISL) + compile & time it — single-statement, sub-second, no LLM |
| `optimize(kernel, backend=mock, iters, candidates, moa, aggregator)` | run the agent loop, return best measured speedup + schedule. `mock` is offline/keyless; `gemini`/`local` drive real models |

## Claude Code

```bash
claude mcp add compilot -e COMPILOT_PYTHON=$HOME/.venvs/compilot/bin/python \
  -- npx -y @cluster2600/compilot-mcp
```

## Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.compilot]
command = "npx"
args = ["-y", "@cluster2600/compilot-mcp"]
env = { COMPILOT_PYTHON = "/Users/you/.venvs/compilot/bin/python" }
```

## Try it

> "List the compilot kernels, then check the legality of `reorder(i, k, j)` on gemm."

`check_legality` returns the ISL verdict and the measured speedup; `optimize`
(default `mock` backend) runs the full search offline.

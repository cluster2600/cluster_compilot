#!/usr/bin/env node
// Thin launcher: the cluster_compilot MCP server is Python (islpy + clang live
// there), so this just execs `python -m compilot.mcp_server` and passes stdio
// straight through — the MCP stdio transport is newline-delimited JSON-RPC, so
// `stdio: "inherit"` is the whole bridge.
//
// Python resolution order:
//   1. $COMPILOT_PYTHON                       (point this at your venv)
//   2. ./.venv/bin/python                     (launched from a checkout)
//   3. python3 / python                       (must have compilot installed)

const { spawn } = require("child_process");
const { existsSync } = require("fs");
const { join } = require("path");

function findPython() {
  if (process.env.COMPILOT_PYTHON) return process.env.COMPILOT_PYTHON;
  const venv = join(
    process.cwd(),
    ".venv",
    process.platform === "win32" ? "Scripts" : "bin",
    process.platform === "win32" ? "python.exe" : "python"
  );
  if (existsSync(venv)) return venv;
  return process.platform === "win32" ? "python" : "python3";
}

const py = findPython();
const child = spawn(py, ["-m", "compilot.mcp_server", ...process.argv.slice(2)], {
  stdio: "inherit",
});

child.on("error", (e) => {
  console.error(`compilot-mcp: failed to launch '${py}': ${e.message}`);
  console.error(
    "Set COMPILOT_PYTHON to your cluster_compilot venv python, or run " +
      "`pip install -e .` in a checkout so `compilot` is importable."
  );
  process.exit(1);
});
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code === null ? 1 : code);
});

#!/usr/bin/env node
// Thin launcher: the cluster_compilot MCP server is Python (islpy + clang live
// there), so this just execs `python -m compilot.mcp_server` and passes stdio
// straight through — the MCP stdio transport is newline-delimited JSON-RPC, so
// `stdio: "inherit"` is the whole bridge.
//
// Python resolution order:
//   1. $COMPILOT_PYTHON                       (point this at your venv)
//   2. python3 / python                       (must have compilot installed)
//
// Nothing is resolved from the working directory: MCP clients start this server
// in whatever project is open, so a ./.venv or ./compilot/ there must not be able
// to substitute its own interpreter or package. Hence no ./.venv lookup, and `-P`
// (Python 3.11+) keeps the cwd off sys.path for `-m`.

const { spawn } = require("child_process");

function findPython() {
  if (process.env.COMPILOT_PYTHON) return process.env.COMPILOT_PYTHON;
  return process.platform === "win32" ? "python" : "python3";
}

const py = findPython();
const child = spawn(py, ["-P", "-m", "compilot.mcp_server", ...process.argv.slice(2)], {
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

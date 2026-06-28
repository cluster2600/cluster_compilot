"""MCP server self-check — protocol handshake + the three tools, no API key.

Drives compilot.mcp_server.handle() with JSON-RPC messages exactly as a client
(Claude Code / Codex) would, and runs a real check_legality through ISL + clang.

    python3 -m tests.test_mcp
"""
import json

from compilot.mcp_server import handle


def test_handshake_and_tools_list():
    init = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-06-18"}})
    assert init["result"]["serverInfo"]["name"] == "compilot", init
    assert "tools" in init["result"]["capabilities"], init
    # notifications/initialized has no id -> no response
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tl = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {t["name"] for t in tl["result"]["tools"]}
    assert names == {"list_kernels", "check_legality", "optimize"}, names
    print(f"OK: handshake + tools/list -> {sorted(names)}")


def test_check_legality_tool():
    r = handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "check_legality",
                           "arguments": {"kernel": "gemm", "schedule": "reorder(i, k, j)"}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["status"] == "success", payload
    assert payload["speedup"] and payload["speedup"] > 1.0, payload
    print(f"OK: check_legality gemm reorder(i,k,j) -> {payload['speedup']:.2f}x")


def test_check_legality_rejects_multistatement():
    r = handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "check_legality",
                           "arguments": {"kernel": "2mm", "schedule": "reorder(i, k, j)"}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert "error" in payload, payload          # multi-statement -> clear redirect, not a crash
    print("OK: check_legality on a multi-statement kernel returns a clear error")


def test_optimize_mock_tool():
    r = handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "optimize",
                           "arguments": {"kernel": "gemm", "backend": "mock", "iters": 6}}})
    payload = json.loads(r["result"]["content"][0]["text"])
    assert payload["best_speedup"] >= 1.0, payload
    assert payload["schedule"], payload
    print(f"OK: optimize(gemm, mock) -> {payload['best_speedup']:.2f}x")


def test_unknown_tool_is_error_not_crash():
    r = handle({"jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "nope", "arguments": {}}})
    assert r["result"]["isError"] is True, r
    # unknown *method* -> JSON-RPC error object
    r2 = handle({"jsonrpc": "2.0", "id": 7, "method": "bogus/method"})
    assert r2["error"]["code"] == -32601, r2
    print("OK: unknown tool -> isError; unknown method -> -32601")


if __name__ == "__main__":
    test_handshake_and_tools_list()
    test_check_legality_tool()
    test_check_legality_rejects_multistatement()
    test_optimize_mock_tool()
    test_unknown_tool_is_error_not_crash()
    print("test_mcp: all checks passed")

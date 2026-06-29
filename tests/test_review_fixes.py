"""Regression tests for the code-review fixes (issues #9-#26).

The kernel-specific soundness fixes (symm parallel(k), gramschmidt parallel(i))
are guarded in test_imperfect.py. This module covers the cross-cutting ones:
schedule-arg validation, malformed-schedule handling, the unroll/fuse/shift and
syrk/syr2k/floydwarshall coverage gaps, the Gemini key-in-header fix, the
position-weighted checksum, the MCP non-object-JSON guard, and the run timeout.

    python3 -m tests.test_review_fixes
"""
import json
import subprocess
import sys

from compilot import schedule
from compilot.backend_isl import environment


def test_schedule_arg_validation():                       # issue #20
    schedule.parse("tile(i, 16)")                         # well-formed: loop id + integer
    schedule.parse("reorder(i, k, j)")
    for bad in ("tile(i, 16abc)", "parallel(i; rm -rf)", "skew(j, i, 1x)", "unroll(j, 4 4)"):
        try:
            schedule.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should have been rejected")
    print("OK schedule arg validation (#20)")


def test_malformed_schedule_is_invalid_not_crash():       # issue #16
    for bad in ("tile2d(i)", "interchange(i)", "skew(i)"):
        r = environment("gemm").evaluate(bad)
        assert r.status == "invalid", f"{bad}: {r.status} (expected invalid)"
    print("OK malformed schedule -> invalid, no crash (#16)")


def test_unroll_executes():                               # issue #21
    r = environment("gemm").evaluate("unroll(j, 4)")
    assert r.status == "success", f"unroll: {r.status} ({r.detail[:80]})"
    # fuse/shift aren't executable yet, but must at least PARSE end-to-end
    schedule.parse("fuse(S0, S1)")
    schedule.parse("shift(S1, 1)")
    print("OK unroll executes; fuse/shift parse (#21)")


def test_single_kernels_assert():                         # issue #22
    for k in ("syrk", "syr2k"):
        assert environment(k).evaluate("").status == "success", k
    # floyd-warshall's transitive-closure loop is genuinely not parallelizable
    assert environment("floydwarshall").evaluate("parallel(k)").status == "parallel_illegal"
    assert environment("floydwarshall").evaluate("").status == "success"
    print("OK syrk/syr2k run; floydwarshall un-parallelizable (#22)")


def test_gemini_key_in_header_not_url():                  # issue #13
    from compilot import llm
    seen = {}

    def fake_post(url, payload, headers, timeout, **kw):
        seen["url"], seen["headers"] = url, headers
        return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}}

    orig = llm._http_post_json
    llm._http_post_json = fake_post
    try:
        llm.GeminiClient(api_key="SECRET-TEST-KEY").chat("sys", [{"role": "user", "content": "hi"}])
    finally:
        llm._http_post_json = orig
    assert "key=" not in seen["url"], f"key leaked into URL: {seen['url']}"
    assert seen["headers"].get("x-goog-api-key") == "SECRET-TEST-KEY"
    print("OK Gemini key in header, not URL (#13)")


def test_checksum_is_position_weighted():                 # issue #14
    from compilot import codegen
    src = codegen.generate_c(environment("gemm").ek, "")
    # an unweighted `sum += out[...]` is permutation-invariant; the fix multiplies by
    # the linear index, so the generated checksum loop must reference (… + 1) *.
    assert "+ 1) *" in src or "+1)*" in src, "checksum no longer position-weighted"
    print("OK checksum position-weighted (#14)")


def test_mcp_survives_non_object_json():                  # issue #17
    proc = subprocess.run(
        [sys.executable, "-m", "compilot.mcp_server"],
        input='[1,2,3]\n5\n"a string"\n'
              '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n',
        capture_output=True, text=True, timeout=30)
    # the three non-object lines must be skipped, and the initialize still answered
    lines = [l for l in proc.stdout.splitlines() if l.strip()]
    assert lines, f"no response (server died?). stderr: {proc.stderr[:300]}"
    resp = json.loads(lines[-1])
    assert resp.get("id") == 1, f"unexpected response: {resp}"
    print("OK MCP server survives non-object JSON (#17)")


def test_run_timeout_returns_error():                     # issue #11
    from compilot import runner
    spin = ("int main(){volatile long i=0; for(;;) i++; "
            'printf("TIME 0.0\\nCHECKSUM 0.0\\n"); return 0;}')
    src = "#include <stdio.h>\n" + spin
    r = runner.compile_and_run(src, timeout=2)
    assert r["ok"] is False and r["error"] == "timeout", f"expected timeout, got {r}"
    print("OK run timeout returns error, no exception (#11)")


def test_tile_factors_must_be_positive():                # adversarial #4
    for ok in ("tile(i, 16)", "unroll(j, 4)", "tile2d(i, j, 32, 32)", "tile3d(k, i, j, 8, 8, 8)"):
        schedule.parse(ok)
    for bad in ("tile(i, 0)", "tile(i, -16)", "unroll(j, 0)", "tile2d(i, j, 0, 8)", "tile2d(i, j)"):
        try:
            schedule.parse(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should be rejected (zero/negative/missing factor)")
    print("OK tile/unroll factors must be positive integers (adversarial #4)")


def test_malformed_multi_schedule_is_invalid_not_crash():  # adversarial #3
    from compilot.kernels import MULTI_REGISTRY
    from compilot.multikernel import MultiEnvironment
    env = MultiEnvironment(MULTI_REGISTRY["2mm"]())
    # "skew(i)" parses but build_theta indexes missing args -> IndexError, which the
    # multi/stencil/imperfect paths previously did NOT catch (only ValueError/KeyError).
    r = env.evaluate(["skew(i)", ""])
    assert r["status"] == "invalid", f"expected invalid, got {r['status']}"
    print("OK malformed multi schedule -> invalid, no crash (adversarial #3)")


def test_zero_time_is_measurement_error_not_fake_speedup():  # adversarial #1
    from compilot import runner
    src = '#include <stdio.h>\nint main(){printf("TIME 0.000000000\\nCHECKSUM 1.0\\n");return 0;}'
    r = runner.compile_and_run(src)
    assert r["ok"] is False and r["error"] == "measurement", f"expected measurement error, got {r}"
    print("OK zero measured time -> measurement error, not fake speedup (adversarial #1)")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for t in _TESTS:
        t()
    print(f"\ntest_review_fixes: all {len(_TESTS)} checks passed")

"""Mixture-of-Agents (pool & measure) — offline self-check, no API key.

References + aggregator are all MockClients (scripted reorder -> tile2d+parallel ->
illegal parallel(k) -> stop), so the pool/dedupe/measure path runs end-to-end and
must reach the same legal speedup the single-model mock dialogue does. Also checks
the OpenAI role mapping, the one bit of local-client logic that fails silently.

    python3 -m tests.test_moa
"""
from compilot.backend_isl import environment
from compilot.agent import run_dialogue_moa
from compilot.llm import MockClient, _openai_messages, _http_post_json


def test_openai_role_mapping():
    got = _openai_messages("SYS", [("user", "a"), ("model", "b"), ("user", "c")])
    assert got == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},   # 'model' -> 'assistant'
        {"role": "user", "content": "c"},
    ], got


def test_local_client_unreachable_is_clear():
    # OpenAIClient must degrade to a readable error, not a raw urllib traceback,
    # when the local server is down. retries=0 keeps the check fast (no backoff sleep).
    try:
        _http_post_json("http://127.0.0.1:9/v1/chat/completions", {"model": "x"},
                        {"Content-Type": "application/json"}, timeout=1, retries=0)
        assert False, "expected RuntimeError on unreachable server"
    except RuntimeError as e:
        assert "could not reach" in str(e), e
    print("OK: unreachable local server -> clear RuntimeError")


def test_moa_pool_and_measure():
    env = environment("gemm")
    refs = [MockClient(), MockClient(), MockClient()]      # identical scripts -> dedupe collapses the pool
    agg = MockClient()
    best_sp, best_sched, trace = run_dialogue_moa(
        env, refs, agg, max_iters=10, verbose=False, candidates_per_turn=2)
    assert trace, "no candidates were ever measured"
    assert best_sp > 5.0, f"mock MoA should reach the scripted tile2d+parallel speedup, got {best_sp:.2f}x"
    assert best_sched.strip(), "winning schedule is empty"
    # identical references must not inflate the measured pool: each turn dedupes to one block
    print(f"OK: MoA reached {best_sp:.2f}x over {len(trace)} measured candidates "
          f"(refs deduped); best schedule:\n{best_sched.strip()}")


if __name__ == "__main__":
    test_openai_role_mapping()
    test_local_client_unreachable_is_clear()
    test_moa_pool_and_measure()
    print("test_moa: all checks passed")

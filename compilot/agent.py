"""The ComPilot agent: the optimization dialogue.

run_dialogue drives one LLM<->compiler conversation (the paper's iterative
optimization phase): present the nest, parse each <schedule> proposal, evaluate
it in the environment, feed the outcome back, keep the best legal speedup, stop
on the stop-token / iteration cap. best_of_k repeats for multi-run (Fig. RQ9).
"""
import re
from concurrent.futures import ThreadPoolExecutor

from . import feedback as _feedback
from . import prompt as _prompt

_SCHED = re.compile(r"<schedule>(.*?)</schedule>", re.S)


def parse_response(text):
    """Return (schedule_text_or_None, stop_requested)."""
    blocks = _SCHED.findall(text)
    if not blocks:
        return None, False
    body = blocks[-1].strip()
    if _prompt.STOP_TOKEN in body:
        return None, True
    return body, False


def run_dialogue(env, llm, max_iters=30, verbose=True, candidates_per_turn=1, tag=""):
    """One dialogue. Returns (best_speedup, best_schedule, trace).

    candidates_per_turn > 1 switches to a fan-out dialogue where the model proposes
    several schedules per turn that are compiled and measured in parallel (see
    _run_dialogue_candidates). tag prefixes log lines so the interleaved output of
    concurrent best-of-k runs stays readable.
    """
    if candidates_per_turn > 1:
        return _run_dialogue_candidates(env, llm, max_iters, verbose, candidates_per_turn, tag)
    lead = f"{tag}  " if tag else "  "
    messages = [("user", _prompt.kernel_message(env))]
    best_sp, best_sched = 1.0, ""
    trace = []
    for it in range(max_iters):
        resp = llm.chat(_prompt.SYSTEM, messages)
        messages.append(("model", resp))
        sched, stop = parse_response(resp)
        if stop:
            if verbose:
                print(f"{lead}iter {it}: agent stopped")
            break
        if sched is None:
            messages.append(("user", "No <schedule> block found. Provide exactly one." + _feedback._CONTINUE))
            continue
        result = env.evaluate(sched)
        trace.append((sched, result.status, result.speedup))
        if result.status == "success" and result.speedup > best_sp:
            best_sp, best_sched = result.speedup, sched
        if verbose:
            sp = f"{result.speedup:.2f}x" if result.speedup else "  -  "
            print(f"{lead}iter {it}: [{result.status:16}] {sp:>7}  best={best_sp:.2f}x")
        messages.append(("user", _feedback.format_feedback(result, best_sp)))
    return best_sp, best_sched, trace


def _run_dialogue_candidates(env, llm, max_iters, verbose, n, tag):
    """Fan-out dialogue: each turn the model proposes up to n schedules, which are
    evaluated in parallel (ThreadPoolExecutor). env.evaluate serializes its own
    polyhedral section; the clang compile/run of each candidate runs concurrently.
    Returns (best_speedup, best_schedule, trace)."""
    lead = f"{tag}  " if tag else "  "
    messages = [("user", _prompt.kernel_message(env) + _prompt.multi_candidate_hint(n))]
    best_sp, best_sched = 1.0, ""
    trace = []
    for it in range(max_iters):
        resp = llm.chat(_prompt.SYSTEM, messages)
        messages.append(("model", resp))
        blocks = [b.strip() for b in _SCHED.findall(resp)]
        if any(_prompt.STOP_TOKEN in b for b in blocks):
            if verbose:
                print(f"{lead}iter {it}: agent stopped")
            break
        if not blocks:
            messages.append(("user", f"No <schedule> block found. Provide 1 to {n}." + _feedback._CONTINUE))
            continue
        cands = blocks[-n:]                       # the latest (up to) n proposals
        with ThreadPoolExecutor(max_workers=len(cands)) as ex:
            results = list(ex.map(env.evaluate, cands))
        for sched, result in zip(cands, results):
            trace.append((sched, result.status, result.speedup))
            if result.status == "success" and result.speedup > best_sp:
                best_sp, best_sched = result.speedup, sched
        if verbose:
            summ = "  ".join(
                (f"{r.speedup:.2f}x" if r.status == "success" and r.speedup else r.status)
                for r in results)
            print(f"{lead}iter {it}: {len(cands)} cand [{summ}]  best={best_sp:.2f}x")
        messages.append(("user", _feedback.format_candidates_feedback(cands, results, best_sp)))
    return best_sp, best_sched, trace


def run_dialogue_moa(env, references, aggregator, max_iters=30, verbose=True,
                     candidates_per_turn=2, max_candidates=8, tag=""):
    """Mixture-of-Agents dialogue (pool & measure), Hermes-style two-tier proposing.

    Each turn: every *reference* model proposes schedules in parallel (advisory),
    the *aggregator* model then synthesizes its own proposals informed by theirs,
    and ALL pooled+deduped candidates are compiled and measured by the ground-truth
    oracle in parallel. The best measured speedup wins; the measured results feed
    back to every agent next turn. The aggregator's turn is the canonical transcript
    entry (references are re-run fresh each turn, like Hermes references).

    references: list of LLM clients (proposers). aggregator: one LLM client.
    Returns (best_speedup, best_schedule, trace).
    """
    lead = f"{tag}  " if tag else "  "
    messages = [("user", _prompt.kernel_message(env) + _prompt.multi_candidate_hint(candidates_per_turn))]
    best_sp, best_sched = 1.0, ""
    trace = []
    for it in range(max_iters):
        with ThreadPoolExecutor(max_workers=len(references)) as ex:   # references propose in parallel
            ref_resps = list(ex.map(lambda c: c.chat(_prompt.SYSTEM, messages), references))
        ref_scheds = [b.strip() for resp in ref_resps for b in _SCHED.findall(resp)]
        agg_msgs = messages + [("user", _prompt.moa_aggregator_hint(ref_scheds, candidates_per_turn))]
        agg_resp = aggregator.chat(_prompt.SYSTEM, agg_msgs)
        agg_scheds = [b.strip() for b in _SCHED.findall(agg_resp)]
        if all(_prompt.STOP_TOKEN in r for r in ref_resps + [agg_resp]):
            if verbose:
                print(f"{lead}iter {it}: all agents stopped")
            break
        pool, seen = [], set()                          # dedupe, drop empty / stop-token blocks
        for s in ref_scheds + agg_scheds:
            if not s or _prompt.STOP_TOKEN in s or s in seen:
                continue
            seen.add(s)
            pool.append(s)
        messages.append(("model", agg_resp))
        if not pool:
            messages.append(("user", f"No <schedule> block found across agents. Propose 1 to "
                                      f"{candidates_per_turn} each." + _feedback._CONTINUE))
            continue
        if len(pool) > max_candidates:
            if verbose:
                print(f"{lead}iter {it}: pooled {len(pool)}, measuring first {max_candidates} "
                      f"(dropped {len(pool) - max_candidates})")    # ponytail: hard cap on compiles/turn
            pool = pool[:max_candidates]
        with ThreadPoolExecutor(max_workers=len(pool)) as ex:        # measure the whole pool in parallel
            results = list(ex.map(env.evaluate, pool))
        for sched, result in zip(pool, results):
            trace.append((sched, result.status, result.speedup))
            if result.status == "success" and result.speedup > best_sp:
                best_sp, best_sched = result.speedup, sched
        if verbose:
            summ = "  ".join(
                (f"{r.speedup:.2f}x" if r.status == "success" and r.speedup else r.status)
                for r in results)
            print(f"{lead}iter {it}: {len(references)} refs -> {len(pool)} cand [{summ}]  best={best_sp:.2f}x")
        messages.append(("user", _feedback.format_candidates_feedback(pool, results, best_sp)))
    return best_sp, best_sched, trace


def _multi_feedback(r, best):
    s = r["status"]
    if s == "success":
        return (f"Legal. Combined speedup {r['speedup']:.2f}x ({r.get('detail','')}). "
                f"Best so far {max(r['speedup'], best):.2f}x." + _feedback._CONTINUE)
    if s in ("illegal", "parallel_illegal"):
        return (f"Illegal schedule for statement writing `{r.get('stmt','?')}`: {s}. "
                f"Fix that statement's schedule." + _feedback._CONTINUE)
    return f"{s}: {r.get('detail','')}" + _feedback._CONTINUE


def run_dialogue_multi(menv, llm, max_iters=30, verbose=True):
    """Dialogue for a multi-statement kernel: one <schedule> block per statement."""
    n = len(menv.mk.statements)
    messages = [("user", _prompt.kernel_message_multi(menv))]
    best_sp, best = 1.0, None
    for it in range(max_iters):
        resp = llm.chat(_prompt.SYSTEM, messages)
        messages.append(("model", resp))
        blocks = [b.strip() for b in _SCHED.findall(resp)]
        if any(_prompt.STOP_TOKEN in b for b in blocks):
            break
        if not blocks:
            messages.append(("user", f"Provide {n} <schedule> blocks." + _feedback._CONTINUE))
            continue
        scheds = (blocks[-n:] if len(blocks) >= n else blocks + [""] * (n - len(blocks)))
        r = menv.evaluate(scheds)
        if r["status"] == "success" and r["speedup"] > best_sp:
            best_sp, best = r["speedup"], scheds
        if verbose:
            sp = f"{r['speedup']:.2f}x" if r.get("speedup") else "  -  "
            print(f"  iter {it}: [{r['status']:16}] {sp:>7}  best={best_sp:.2f}x")
        messages.append(("user", _multi_feedback(r, best_sp)))
    return best_sp, best


def best_of_k(env, make_llm, K=5, max_iters=30, verbose=True,
              candidates_per_turn=1, max_workers=None):
    """Run K independent dialogues concurrently; return the best plus per-run results.

    The runs are I/O- and subprocess-bound (LLM calls + clang compile/run, both of
    which release the GIL), so a thread pool gives near-linear wall-clock speedup
    even on the standard GIL-enabled interpreter. Each run gets its own LLM client
    via make_llm(); the shared environment's polyhedral section is lock-guarded in
    env.evaluate, and the baseline is pre-warmed once before fan-out.
    """
    env.baseline()                                # compile the shared baseline before fan-out
    def _one(k):
        sp, sched, _ = run_dialogue(env, make_llm(), max_iters, verbose,
                                    candidates_per_turn, tag=f"[run {k + 1}/{K}]")
        return sp, sched
    with ThreadPoolExecutor(max_workers=max_workers or K) as ex:
        runs = list(ex.map(_one, range(K)))
    best = max(runs, key=lambda r: r[0])
    return best, runs

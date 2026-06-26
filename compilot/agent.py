"""The ComPilot agent: the optimization dialogue.

run_dialogue drives one LLM<->compiler conversation (the paper's iterative
optimization phase): present the nest, parse each <schedule> proposal, evaluate
it in the environment, feed the outcome back, keep the best legal speedup, stop
on the stop-token / iteration cap. best_of_k repeats for multi-run (Fig. RQ9).
"""
import re

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


def run_dialogue(env, llm, max_iters=30, verbose=True):
    """One dialogue. Returns (best_speedup, best_schedule, trace)."""
    messages = [("user", _prompt.kernel_message(env))]
    best_sp, best_sched = 1.0, ""
    trace = []
    for it in range(max_iters):
        resp = llm.chat(_prompt.SYSTEM, messages)
        messages.append(("model", resp))
        sched, stop = parse_response(resp)
        if stop:
            if verbose:
                print(f"  iter {it}: agent stopped")
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
            print(f"  iter {it}: [{result.status:16}] {sp:>7}  best={best_sp:.2f}x")
        messages.append(("user", _feedback.format_feedback(result, best_sp)))
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


def best_of_k(env, make_llm, K=5, max_iters=30, verbose=True):
    """Run K independent dialogues; return the best plus per-run results."""
    runs = []
    for k in range(K):
        if verbose:
            print(f"--- run {k + 1}/{K} ---")
        sp, sched, _ = run_dialogue(env, make_llm(), max_iters, verbose)
        runs.append((sp, sched))
    best = max(runs, key=lambda r: r[0])
    return best, runs

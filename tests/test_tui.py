"""TUI worker dispatch — single, multi, and MoA-multi paths all feed on_eval. No key.

Drives compilot.tui._worker (the headless half of the monitor; no curses needed)
and checks each kernel kind streams measured candidates into the shared State, so
the live leaderboard has something to render for multi-statement + MoA fan-out.

    python3 -m tests.test_tui
"""
from types import SimpleNamespace

from compilot.tui import _State, _worker


def _run(**kw):
    args = SimpleNamespace(kernel="gemm", backend="mock", model="x", iters=6,
                           candidates=3, base_url="", moa="", aggregator="")
    args.__dict__.update(kw)
    st = _State()
    _worker(st, args)
    assert st.done and st.error is None, st.error
    assert st.evals, f"no candidates captured for kernel={args.kernel} moa={args.moa!r}"
    return st


def test_single_statement():
    st = _run(kernel="gemm")
    print(f"OK: single gemm -> {len(st.evals)} evals, best {st.best[0]:.2f}x")


def test_multi_statement():
    st = _run(kernel="2mm")
    assert st.best[1], "multi best schedule label is empty"
    print(f"OK: multi 2mm -> {len(st.evals)} evals, best {st.best[0]:.2f}x")


def test_moa_multi_fanout():
    st = _run(kernel="2mm", moa="mock,mock", aggregator="mock")
    print(f"OK: MoA-multi 2mm -> {len(st.evals)} evals, best {st.best[0]:.2f}x")


if __name__ == "__main__":
    test_single_statement()
    test_multi_statement()
    test_moa_multi_fanout()
    print("test_tui: all checks passed")

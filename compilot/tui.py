"""TUI monitor for the ComPilot search — live leaderboard + falling sakura. 🌸

Runs a single-statement dialogue in a worker thread and renders, via stdlib
curses (no Textual/rich dep), the candidates as they're measured plus an ASCII
Japanese animation: drifting cherry-blossom petals and a waving maneki-neko (the
lucky cat waves once per new best speedup it brings in).

  python -m compilot.tui --kernel gemm            (or the `compilot-tui` script)
  python -m compilot.tui --kernel syrk --candidates 4 --iters 15

Mock backend by default (offline, no key). Single-statement kernels only — the
live hook lives on run_dialogue; multi-statement/stencil go through run_agent.py.

# ponytail: curses + a shared list under a lock. No event bus, no async; the
# worker only touches the list, only the main thread touches the screen.
"""
import curses
import locale
import random
import threading

locale.setlocale(locale.LC_ALL, "")

PETALS = "❀✿❁✾❃･｡"
HEADER = "❀ ComPilot 最適化モニター ❀"      # "optimization monitor"

CAT = [                                       # 2 frames; the paw (\ / ) waves
    [r"  /\_/\  ", r" ( ^.^ )/", r"  c(\")(\")"],
    [r" \/\_/\  ", r" ( ^.^ ) ", r"  c(\")(\")"],
]


class _State:
    def __init__(self):
        self.lock = threading.Lock()
        self.evals = []          # (sched, status, speedup)
        self.best = (1.0, "")
        self.done = False
        self.error = None


def _worker(state, args):
    try:
        from .agent import (run_dialogue, run_dialogue_multi,
                            run_dialogue_moa, run_dialogue_moa_multi)
        from .backend_isl import environment
        from .kernels import MULTI_REGISTRY, STENCIL_REGISTRY, sized_kernel
        from .mcp_server import _make_client
        from .multikernel import MultiEnvironment
        from .stencil import StencilEnvironment

        def on_eval(label, status, speedup):
            with state.lock:
                state.evals.append((label, status, speedup))

        base_spec = "mock" if args.backend == "mock" else f"{args.backend}:{args.model}"
        refs = [_make_client(s.strip(), args.base_url, 0.9)
                for s in args.moa.split(",") if s.strip()] if args.moa else []
        agg = _make_client(args.aggregator or base_spec, args.base_url, 0.4) if args.moa else None

        if args.kernel in MULTI_REGISTRY or args.kernel in STENCIL_REGISTRY:   # >1 statement
            menv = (StencilEnvironment(sized_kernel(args.kernel, args.size)) if args.kernel in STENCIL_REGISTRY
                    else MultiEnvironment(sized_kernel(args.kernel, args.size)))
            if args.moa:
                sp, best = run_dialogue_moa_multi(menv, refs, agg, max_iters=args.iters,
                                                  verbose=False, on_eval=on_eval)
            else:
                sp, best = run_dialogue_multi(menv, _make_client(base_spec, args.base_url),
                                              max_iters=args.iters, verbose=False, on_eval=on_eval)
            sched = " | ".join(f"[{i}] {s.strip() or 'id'}" for i, s in enumerate(best or []))
        else:                                                                 # single statement
            env = environment(args.kernel, args.size)
            if args.moa:
                sp, sched, _ = run_dialogue_moa(env, refs, agg, max_iters=args.iters, verbose=False,
                                                candidates_per_turn=args.candidates, on_eval=on_eval)
            else:
                sp, sched, _ = run_dialogue(env, _make_client(base_spec, args.base_url),
                                            max_iters=args.iters, verbose=False,
                                            candidates_per_turn=args.candidates, on_eval=on_eval)
        with state.lock:
            state.best = (sp, sched)
    except Exception as e:                    # show it in the panel, don't crash curses
        with state.lock:
            state.error = str(e)
    finally:
        with state.lock:
            state.done = True


def _put(win, y, x, s, attr=0):
    """addstr that clips to the window and never raises at the edges."""
    h, w = win.getmaxyx()
    if not (0 <= y < h) or x >= w:
        return
    if x < 0:
        s, x = s[-x:], 0
    try:
        win.addstr(y, x, s[:max(0, w - x - 1)], attr)
    except curses.error:
        pass


def _main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(90)                        # ~11 fps
    pink = cyan = green = dim = 0
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_MAGENTA, -1)
        curses.init_pair(2, curses.COLOR_CYAN, -1)
        curses.init_pair(3, curses.COLOR_GREEN, -1)
        curses.init_pair(4, 8, -1)
        pink, cyan, green, dim = (curses.color_pair(i) for i in (1, 2, 3, 4))

    state = _State()
    threading.Thread(target=_worker, args=(state, args), daemon=True).start()

    h, w = stdscr.getmaxyx()
    petals = [[random.uniform(0, h), random.randrange(max(1, w)),
               random.choice(PETALS), random.uniform(0.3, 1.0)] for _ in range(max(6, w // 5))]
    frame = 0
    last_best = 1.0
    wave_until = 0

    while True:
        ch = stdscr.getch()
        if ch in (ord("q"), ord("Q")):
            break
        if ch == curses.KEY_RESIZE:
            h, w = stdscr.getmaxyx()
        stdscr.erase()

        # falling sakura
        for p in petals:
            p[0] += p[3]
            if p[0] >= h:
                p[0], p[1], p[2] = 0, random.randrange(max(1, w)), random.choice(PETALS)
            _put(stdscr, int(p[0]), p[1], p[2], pink)

        # header
        _put(stdscr, 0, max(0, (w - len(HEADER)) // 2), HEADER, cyan | curses.A_BOLD)

        with state.lock:
            evals = list(state.evals)
            done, error, best = state.done, state.error, state.best
        n = len(evals)
        legal = sorted((e for e in evals if e[1] == "success" and e[2]),
                       key=lambda e: -e[2])
        cur_best = legal[0][2] if legal else 1.0
        if cur_best > last_best + 1e-9:       # new record → cat waves for ~0.7s
            last_best, wave_until = cur_best, frame + 8

        # leaderboard
        _put(stdscr, 2, 2, "── 速度ランキング (top speedups) ──", cyan)
        if not legal:
            _put(stdscr, 4, 4, "measuring…" if not done else "no legal schedule found", dim)
        for i, (sched, _st, sp) in enumerate(legal[:6]):
            label = " ; ".join(ln.strip() for ln in sched.strip().splitlines() if ln.strip()) or "(identity)"
            _put(stdscr, 4 + i, 4, f"{i + 1}. {sp:6.2f}x  {label}", green if i == 0 else 0)

        # waving maneki-neko, bottom-left
        cat = CAT[1 if frame < wave_until and (frame // 2) % 2 else 0]
        for i, row in enumerate(cat):
            _put(stdscr, h - 5 + i, 2, row.replace('\\"', '"'), 0)

        # status
        if error:
            status = f"⚠ {error[:w - 6]}"
            attr = curses.A_BOLD
        elif done:
            sp, _sched = best
            status = f"完了 DONE — best {sp:.2f}x over {n} candidates   press q"
            attr = green | curses.A_BOLD
        else:
            mode = f"MoA×{len([s for s in args.moa.split(',') if s.strip()])}" if args.moa else args.backend
            status = (f"kernel={args.kernel}  {mode}  "
                      f"evals={n}  best={cur_best:.2f}x   [q]uit")
            attr = 0
        _put(stdscr, h - 1, 0, status.ljust(w - 1), attr)

        stdscr.refresh()
        frame += 1
        if done and ch != -1:                 # let the user read the final frame, then any key quits
            break


def run():
    import argparse
    ap = argparse.ArgumentParser(description="Live TUI monitor for the ComPilot search (falling sakura 🌸).")
    ap.add_argument("--kernel", default="gemm", help="single-statement, multi-statement, or stencil")
    ap.add_argument("--backend", default="mock", choices=["mock", "gemini", "local"])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=3, help="parallel candidates/turn (single-statement)")
    ap.add_argument("--moa", default="", help="watch Mixture-of-Agents fan-out: comma-separated "
                                              "reference specs, e.g. 'gemini:gemini-2.5-flash,local:qwen2.5-coder:32b'")
    ap.add_argument("--aggregator", default="", help="MoA aggregator spec (default: backend:model)")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--size", default="LARGE",
                    choices=["MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE"],
                    help="PolyBench dataset size class")
    args = ap.parse_args()

    from .kernels import REGISTRY, MULTI_REGISTRY, STENCIL_REGISTRY, IMPERFECT_REGISTRY
    known = set(REGISTRY) | set(MULTI_REGISTRY) | set(STENCIL_REGISTRY) | set(IMPERFECT_REGISTRY)
    if args.kernel not in known:
        raise SystemExit(f"unknown kernel {args.kernel!r}; choose from {sorted(known)}")
    curses.wrapper(_main, args)


if __name__ == "__main__":
    run()

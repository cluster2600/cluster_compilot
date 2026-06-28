"""Python 3.15 readiness probe.

cluster_compilot's code is already forward-compatible (requires-python >=3.14, no
upper cap; deprecation-clean). The only thing standing between us and 3.15 is the
`islpy` binary wheel — it has no `cp315` build yet, and a source build needs the
ISL C library + a compiler. This script answers, in one command, "is it ready?":

  - does islpy publish a cp315 (and cp315t, for free-threading) wheel yet?
  - does the interpreter you're running satisfy our version guard?

When the cp315 line flips to YES, `pip install -e .` on 3.15 should just work.

    python3 check_py315.py
"""
import json
import re
import sys
import urllib.request

ISLPY_JSON = "https://pypi.org/pypi/islpy/json"


def cp_tags(filenames):
    """The distinct CPython ABI tags (cp310, cp314, cp315t, ...) across wheel names."""
    tags = set()
    for name in filenames:
        if name.endswith(".whl"):
            tags.update(re.findall(r"cp3\d\dt?", name))
    return tags


def main():
    running = sys.version_info
    print(f"running interpreter: {running.major}.{running.minor}  "
          f"(guard requires >=3.14: {'OK' if running >= (3, 14) else 'TOO OLD'})")
    try:
        with urllib.request.urlopen(ISLPY_JSON, timeout=30) as r:
            data = json.load(r)
    except Exception as e:
        print(f"could not reach PyPI: {e}")
        return 2
    files = [f["filename"] for rel in data["releases"].values() for f in rel]
    tags = cp_tags(files)
    has315 = any(t.startswith("cp315") for t in tags)
    has315t = "cp315t" in tags
    latest = data["info"]["version"]
    print(f"islpy latest: {latest}  ({data['info'].get('requires_python')})")
    print(f"islpy cp315 wheel:   {'YES' if has315 else 'NO  (still blocked)'}")
    print(f"islpy cp315t wheel:  {'YES' if has315t else 'NO'}  (needed for free-threaded 3.15t)")
    print()
    if has315:
        print("READY: islpy ships a 3.15 wheel — pip install -e . should work on 3.15.")
        return 0
    print("BLOCKED: waiting on an islpy cp315 wheel. Re-run this when 3.15 is closer to release.")
    return 1


if __name__ == "__main__":
    # offline self-check of the only non-trivial logic
    assert cp_tags(["islpy-2026.1-cp314-cp314-macosx_11_0_arm64.whl",
                    "islpy-2026.1-cp315t-cp315t-manylinux_x86_64.whl",
                    "islpy-2026.1.tar.gz"]) == {"cp314", "cp315t"}, "cp_tags broken"
    sys.exit(main())

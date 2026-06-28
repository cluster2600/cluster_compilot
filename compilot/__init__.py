"""compilot — agentic auto-scheduling (polyhedral legality + LLM loop transforms).

Requires Python 3.14+. The agent fans out across threads (parallel best-of-k and
per-turn candidate schedules); the heavy work — LLM calls and clang compile/run —
is I/O- and subprocess-bound and releases the GIL, so it scales on the standard
interpreter. (islpy ships no free-threaded wheel yet, so the GIL build is used.)
"""
import sys

if sys.version_info < (3, 14):
    raise RuntimeError(
        f"compilot requires Python 3.14+, but is running on "
        f"{sys.version_info.major}.{sys.version_info.minor}."
    )

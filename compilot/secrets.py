"""Fetch secrets from OpenBao at runtime (never persisted to disk or printed).

Mirrors Maxime's OpenBao-first secrets pattern: prefer the GEMINI_API_KEY env
var if already set, otherwise pull `secrets/google` field `api_key` via the bao
CLI (which uses the local addr + saved token). OpenBao must be unsealed.
"""
import os
import subprocess


def gemini_key():
    k = os.environ.get("GEMINI_API_KEY")
    if k:
        return k
    env = dict(os.environ)
    env.setdefault("BAO_ADDR", env.get("VAULT_ADDR", "http://127.0.0.1:8200"))
    try:
        out = subprocess.run(
            ["bao", "kv", "get", "-field=api_key", "secrets/google"],
            capture_output=True, text=True, env=env, timeout=15,
        )
    except FileNotFoundError:
        raise RuntimeError("bao CLI not found and GEMINI_API_KEY unset")
    if out.returncode != 0:
        raise RuntimeError(f"OpenBao read failed (sealed?): {out.stderr.strip()[:200]}")
    key = out.stdout.strip()
    if not key:
        raise RuntimeError("secrets/google api_key was empty")
    return key

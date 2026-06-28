"""LLM clients. Gemini (paper default) over the REST API via stdlib — no SDK to
install. MockClient scripts a dialogue so the whole agent loop is testable with
no key. Both expose .chat(system, messages) and token counters (for RQ2 cost).
"""
import json
import os
import ssl
import time
import urllib.error
import urllib.request

# Key goes in the x-goog-api-key header, NOT the URL: query-string secrets leak
# into proxy/server access logs and into any exception that carries the URL.
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

try:                                    # macOS python.org builds lack a system CA bundle
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


def _http_post_json(url, payload, headers, timeout, retries=2, backoff=0.5):
    """POST JSON and parse the JSON response, retrying transient failures (network
    errors, timeouts, HTTP 5xx) with exponential backoff. HTTP 4xx is NOT retried —
    a bad model name or malformed request won't fix itself — and is surfaced with
    the server's own message. Raises RuntimeError with a readable reason on failure.
    """
    data = json.dumps(payload).encode()
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if e.code < 500:                       # client error (e.g. 404 model not found): don't retry
                raise RuntimeError(f"{url} returned HTTP {e.code}: {detail}") from None
            last = f"HTTP {e.code}: {detail}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = getattr(e, "reason", e)
        if attempt < retries:
            time.sleep(backoff * 2 ** attempt)     # 0.5s, 1s, ...
    raise RuntimeError(f"could not reach {url} after {retries + 1} tries: {last} "
                       f"(is the server running and the model pulled?)")


def _openai_messages(system, messages):
    """Map our (role, text) turns to OpenAI chat format; 'model' -> 'assistant'."""
    out = [{"role": "system", "content": system}]
    for r, t in messages:
        out.append({"role": "assistant" if r == "model" else "user", "content": t})
    return out


class GeminiClient:
    def __init__(self, model="gemini-2.5-flash", api_key=None, temperature=0.7):
        self.model = model
        if api_key:
            self.key = api_key
        else:
            from . import secrets
            self.key = secrets.gemini_key()
        self.temperature = temperature
        self.in_tokens = self.out_tokens = 0

    def chat(self, system, messages):
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": r, "parts": [{"text": t}]} for r, t in messages],
            "generationConfig": {"temperature": self.temperature},
        }
        out = _http_post_json(
            GEMINI_URL.format(model=self.model), body,
            {"Content-Type": "application/json", "x-goog-api-key": self.key}, timeout=180)
        try:                                  # safety-blocked/empty responses lack candidates
            cand = out["candidates"][0]
            text = "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected Gemini response: {str(out)[:300]}") from None
        um = out.get("usageMetadata", {})
        self.in_tokens += um.get("promptTokenCount", 0)
        self.out_tokens += um.get("candidatesTokenCount", 0)
        return text


class OpenAIClient:
    """OpenAI-compatible chat client over stdlib — one client for any local server
    that speaks /v1/chat/completions: Ollama (base_url .../v1), vLLM, NVIDIA NIM,
    LM Studio, llama.cpp. Same .chat(system, messages) + token counters as Gemini.

        OpenAIClient("qwen2.5-coder:32b", base_url="http://localhost:11434/v1")
    """

    def __init__(self, model, base_url="http://localhost:11434/v1", api_key=None, temperature=0.7):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.in_tokens = self.out_tokens = 0

    def chat(self, system, messages):
        body = {"model": self.model,
                "messages": _openai_messages(system, messages),
                "temperature": self.temperature}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        out = _http_post_json(f"{self.base_url}/chat/completions", body,
                              headers, timeout=600)        # local models can be slow
        try:
            text = out["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"unexpected response from {self.base_url}: {str(out)[:300]}") from None
        u = out.get("usage", {})
        self.in_tokens += u.get("prompt_tokens", 0)
        self.out_tokens += u.get("completion_tokens", 0)
        return text


class MockClient:
    """Scripted driver: reorder -> tile2d+parallel -> (illegal) parallel(k) -> stop."""
    SCRIPT = [
        "Reasoning: GEMM; k is the reduction loop. Make k the middle loop for unit-stride "
        "access to B and C.\n<schedule>\nreorder(i, k, j)\n</schedule>",
        "Reasoning: good. Now tile for cache reuse and parallelize the outer tile loop.\n"
        "<schedule>\nreorder(i, k, j)\ntile2d(i, j, 64, 64)\nparallel(i_t)\n</schedule>",
        "Reasoning: can I also parallelize the reduction loop k?\n"
        "<schedule>\nreorder(i, k, j)\ntile2d(i, j, 64, 64)\nparallel(i_t)\nparallel(k)\n</schedule>",
        "Reasoning: k carries the reduction; keep the previous best. Done.\n"
        "<schedule>no_further_transformations</schedule>",
    ]

    def __init__(self):
        self.step = 0
        self.in_tokens = self.out_tokens = 0

    def chat(self, system, messages):
        r = self.SCRIPT[min(self.step, len(self.SCRIPT) - 1)]
        self.step += 1
        return r

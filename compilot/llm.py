"""LLM clients. Gemini (paper default) over the REST API via stdlib — no SDK to
install. MockClient scripts a dialogue so the whole agent loop is testable with
no key. Both expose .chat(system, messages) and token counters (for RQ2 cost).
"""
import json
import os
import ssl
import urllib.request

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

try:                                    # macOS python.org builds lack a system CA bundle
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()


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
        req = urllib.request.Request(
            GEMINI_URL.format(model=self.model, key=self.key),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180, context=_SSL_CTX) as resp:
            out = json.load(resp)
        cand = out["candidates"][0]
        text = "".join(p.get("text", "") for p in cand["content"]["parts"])
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
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=600, context=_SSL_CTX) as resp:  # local models can be slow
            out = json.load(resp)
        text = out["choices"][0]["message"]["content"] or ""
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

"""Standalone JSON connectivity check; the harness uses medadaptagent.knowledge.llm_client."""
import ipaddress
import http.client
import json
import os
import socket
import sys
from urllib import error, parse, request

DEFAULT_URL = "http://127.0.0.1:11434/v1"
DEFAULT_MODEL = "qwen3-vl:4b-instruct"


class LLMError(RuntimeError):
    """An actionable error that does not expose credentials or server bodies."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def call_llm(prompt, *, max_tokens=512, timeout=120):
    """Return validated JSON text from the explicitly configured model.

    Reads MEDADAPT_LLM_BASE_URL, MEDADAPT_LLM_MODEL, and optional
    MEDADAPT_LLM_API_KEY from the environment. No .env files are loaded.
    This transport does not authorize or execute adaptation actions.
    """
    base = os.environ.get("MEDADAPT_LLM_BASE_URL", DEFAULT_URL).strip().rstrip("/")
    model = os.environ.get("MEDADAPT_LLM_MODEL", DEFAULT_MODEL).strip()
    key = os.environ.get("MEDADAPT_LLM_API_KEY", "").strip()
    try:
        parts = parse.urlsplit(base)
        host = parts.hostname
        _ = parts.port  # Reject malformed ports without echoing the URL.
    except ValueError:
        raise LLMError("Invalid MEDADAPT_LLM_BASE_URL.") from None
    local = host == "localhost"
    try:
        local = local or ipaddress.ip_address(host).is_loopback
    except ValueError:
        pass
    if (parts.scheme not in {"http", "https"} or not host
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment or any(c.isspace() for c in base)):
        raise LLMError("Use a base URL without credentials, query parameters, or fragments.")
    if parts.scheme == "http" and not local:
        raise LLMError("Remote LLM endpoints require HTTPS; HTTP is allowed only on loopback.")
    if not model or not isinstance(prompt, str) or not prompt.strip():
        raise LLMError("A model name and nonempty prompt are required.")
    if not isinstance(max_tokens, int) or max_tokens < 1 or timeout <= 0:
        raise LLMError("max_tokens and timeout must be positive.")
    if key and (not key.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise LLMError("The API key contains an invalid header character.")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return one JSON object, with no other text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = request.Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers=headers)
    handlers = [_NoRedirect()]
    if local:
        handlers.append(request.ProxyHandler({}))
    try:
        with request.build_opener(*handlers).open(req, timeout=timeout) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise LLMError("LLM response exceeded the size limit.")
    except error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise LLMError(f"LLM endpoint returned HTTP {code}. Check the endpoint, model, and authentication; response body omitted.") from None
    except (error.URLError, TimeoutError, socket.timeout, OSError, http.client.HTTPException):
        raise LLMError("Cannot reach the configured LLM. Start the server, check its address, and ensure the model is installed.") from None
    try:
        choice = json.loads(raw)["choices"][0]
        if not isinstance(choice, dict):
            raise ValueError
        if choice.get("finish_reason") != "stop":
            raise LLMError("LLM response did not finish normally; no output was accepted. Check model support or increase the token limit.")
        content = choice["message"]["content"]
        if not isinstance(content, str) or not isinstance(json.loads(content), dict):
            raise ValueError
    except (ValueError, KeyError, IndexError, TypeError):
        raise LLMError("LLM returned an invalid response or non-object JSON; no output was accepted.") from None
    return content


def main():
    """Synthetic connectivity check: sends no paper records or medical images."""
    try:
        result = json.loads(call_llm('Return exactly this JSON object: {"status":"ok"}', max_tokens=128))
        if result != {"status": "ok"}:
            raise LLMError("LLM replied, but failed the expected JSON connectivity check.")
    except LLMError as exc:
        print(f"LLM check failed: {exc}", file=sys.stderr)
        return 1
    print("LLM JSON connectivity check passed. No medical data was sent; no adaptation was run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

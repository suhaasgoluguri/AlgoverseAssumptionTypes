"""Probe a chat-completions endpoint for the specific features clients.py relies on.

Standalone and dependency-free (stdlib urllib only) so it works against any
endpoint — including a managed-compute deployment that may not speak the
OpenAI v1 schema at all — without going through the openai SDK, whose error
handling can obscure the raw response body needed to diagnose that.

Checks, independently (one failing does not block the others):
  1. plain completion, with max_completion_tokens vs max_tokens
  2. response_format: {"type": "json_object"}
  3. forced tool_choice naming a specific function

Usage:
    uv run python scripts/probe_endpoint.py \
        --endpoint https://your-endpoint/v1 \
        --api-key $YOUR_API_KEY \
        --deployment your-deployment-or-model-name

    # or rely on AZURE_OPENAI_* / .env to check the current configured deployment:
    uv run python scripts/probe_endpoint.py
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()


def _post(url: str, api_key: str, body: dict, auth_header: str) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if auth_header == "bearer":
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["api-key"] = api_key
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return -1, str(exc.reason)


def _truncate(text: str, limit: int = 600) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _try_both_auth(url: str, api_key: str, body: dict) -> tuple[str, int, str]:
    """Try Authorization: Bearer first (what AsyncOpenAI sends), then api-key."""
    status, text = _post(url, api_key, body, "bearer")
    if status == 401:
        status2, text2 = _post(url, api_key, body, "api-key")
        if status2 != 401:
            return "api-key", status2, text2
    return "bearer", status, text


def probe_plain_completion(url: str, api_key: str, model: str) -> None:
    print("\n=== 1. Plain completion ===")
    base_messages = [{"role": "user", "content": "Reply with exactly: ok"}]
    for token_param in ("max_completion_tokens", "max_tokens"):
        body = {"model": model, "messages": base_messages, token_param: 16}
        auth, status, text = _try_both_auth(url, api_key, body)
        ok = status == 200
        print(f"  {token_param} + auth={auth}: HTTP {status} {'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"    {_truncate(text)}")
        else:
            try:
                content = json.loads(text)["choices"][0]["message"]["content"]
                print(f"    content: {content!r}")
            except (KeyError, IndexError, json.JSONDecodeError):
                print(f"    (200 but unexpected shape) {_truncate(text)}")


def probe_json_object(url: str, api_key: str, model: str) -> None:
    print("\n=== 2. response_format: json_object ===")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with JSON only."},
            {"role": "user", "content": 'Return {"answer": "B", "rationale": "test"}'},
        ],
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }
    auth, status, text = _try_both_auth(url, api_key, body)
    ok = status == 200
    print(f"  auth={auth}: HTTP {status} {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    {_truncate(text)}")
        return
    try:
        content = json.loads(text)["choices"][0]["message"]["content"]
        print(f"    content: {content!r}")
        json.loads(content)
        print("    -> valid JSON body: PASS")
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"    -> did not produce valid JSON: FAIL ({exc})")


def probe_forced_tool_choice(url: str, api_key: str, model: str) -> None:
    print("\n=== 3. Forced tool_choice (specific function) ===")
    tool = {
        "type": "function",
        "function": {
            "name": "record_answer",
            "description": "Record the final answer letter.",
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "string", "enum": ["A", "B", "C"]}},
                "required": ["answer"],
            },
        },
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Pick answer B."}],
        "max_tokens": 64,
        "tools": [tool],
        "tool_choice": {"type": "function", "function": {"name": "record_answer"}},
        "parallel_tool_calls": False,
    }
    auth, status, text = _try_both_auth(url, api_key, body)
    ok = status == 200
    print(f"  auth={auth}: HTTP {status} {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"    {_truncate(text)}")
        return
    try:
        choice = json.loads(text)["choices"][0]
        finish_reason = choice.get("finish_reason")
        tool_calls = choice.get("message", {}).get("tool_calls") or []
        print(f"    finish_reason: {finish_reason!r}, tool_calls: {len(tool_calls)}")
        if finish_reason == "tool_calls" and len(tool_calls) == 1:
            print(f"    -> forced call to {tool_calls[0]['function']['name']!r}: PASS")
        else:
            print("    -> did not force exactly one call as required: FAIL")
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        print(f"    -> unexpected response shape: FAIL ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    args = parser.parse_args()

    if not args.endpoint or not args.api_key or not args.deployment:
        parser.error(
            "endpoint, api-key, and deployment are required "
            "(pass as flags or set AZURE_OPENAI_ENDPOINT / _API_KEY / _DEPLOYMENT)"
        )

    base = args.endpoint.rstrip("/")
    url = f"{base}/chat/completions"
    print(f"Probing {url}  (model={args.deployment!r})")

    probe_plain_completion(url, args.api_key, args.deployment)
    probe_json_object(url, args.api_key, args.deployment)
    probe_forced_tool_choice(url, args.api_key, args.deployment)

    print(
        "\nIf 1 fails: the endpoint likely doesn't speak the OpenAI v1 schema at all — "
        "the AsyncOpenAI client in clients.py can't be pointed at it as-is.\n"
        "If 1 passes only with max_tokens: change max_completion_tokens -> max_tokens "
        "in _base_completion_kwargs().\n"
        "If 2 fails: drop response_format and rely on prompt instructions + parse_or_repair.\n"
        "If 3 fails: switch generate_with_forced_tool to tool_choice='auto' and validate "
        "the model actually calls the tool, or drop forced tool calling for this model."
    )


if __name__ == "__main__":
    main()

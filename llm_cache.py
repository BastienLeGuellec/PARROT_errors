from mistralai import Mistral
import hashlib
import json
import sqlite3
import time
from typing import Any
import openai
from common import canonical_json

# =============================
# Generic cache (payload hashing)
# =============================


def _generic_cache_conn(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS openai_generic_cache (
            endpoint TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            model TEXT,
            url TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (endpoint, url, request_hash)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generic_cache_model ON openai_generic_cache(model)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_generic_cache_url ON openai_generic_cache(url)")
    return conn


def _compute_request_hash(endpoint: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    # Build a minimal, canonical signature across all relevant parameters
    # NOTE: URL is NOT included in hash to preserve existing cache entries
    payload: dict[str, Any] = {"endpoint": endpoint, **kwargs}
    sig = canonical_json(payload)
    return hashlib.sha256(sig.encode("utf-8")).hexdigest(), sig


def _response_to_json_str(resp: Any) -> str:
    # Try the common OpenAI Python SDK methods first, then fall back to JSON
    try:
        return resp.model_dump_json()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return resp.json()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)))
    except:
        pass
    try:
        return canonical_json(resp)
    except Exception:
        return str(resp)


def cached_chat_completions_create(
    client: openai.OpenAI,
    cache_path: str,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generic cache wrapper for client.chat.completions.create.
    - Hashes the full request payload (model, messages, temperature, tools, response_format, etc.).
    - Automatically detects inference URL from client.base_url for tracking.
    - Skips caching when stream=True.
    - Returns a Python dict parsed from the OpenAI response JSON (consistent on cache hit/miss).

    Args:
        client: OpenAI client instance (URL auto-detected from client.base_url)
        cache_path: Path to SQLite cache database
        **kwargs: Arguments passed to chat.completions.create
    """
    # Extract URL from client for tracking (always store full URL)
    client_url = str(client.base_url) if hasattr(
        client, 'base_url') else "https://api.openai.com/v1/"

    if kwargs.get("stream"):
        # Streaming isn't cached here
        resp: Any = client.chat.completions.create(**kwargs)  # type: ignore
        return json.loads(_response_to_json_str(resp))

    endpoint = "chat.completions.create"
    req_hash, req_json = _compute_request_hash(endpoint, kwargs)
    model = kwargs.get("model")

    conn = _generic_cache_conn(cache_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT response_json FROM openai_generic_cache WHERE endpoint=? AND request_hash=? AND url=?",
        (endpoint, req_hash, client_url),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    # Miss: perform call and store
    resp: Any = client.chat.completions.create(**kwargs)  # type: ignore

    resp_json = _response_to_json_str(resp)
    conn.execute(
        "INSERT OR REPLACE INTO openai_generic_cache (endpoint, request_hash, model, url, request_json, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (endpoint, req_hash, model, client_url, req_json, resp_json, time.time()),
    )
    conn.commit()
    conn.close()
    return json.loads(resp_json)


def cached_responses_create(
    client: openai.OpenAI,
    cache_path: str,
    /,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Generic cache wrapper for client.responses.create (Responses API).
    - Hashes the full request payload.
    - Automatically detects inference URL from client.base_url for tracking.
    - Skips caching when stream=True.
    - Returns a Python dict parsed from the OpenAI response JSON.

    Args:
        client: OpenAI client instance (URL auto-detected from client.base_url)
        cache_path: Path to SQLite cache database
        **kwargs: Arguments passed to responses.create
    """
    # Extract URL from client for tracking (always store full URL)
    client_url = str(client.base_url) if hasattr(
        client, 'base_url') else "https://api.openai.com/v1/"

    if kwargs.get("stream"):
        # Streaming isn't cached here
        resp: Any = client.responses.create(**kwargs)  # type: ignore
        return json.loads(_response_to_json_str(resp))

    endpoint = "responses.create"
    req_hash, req_json = _compute_request_hash(endpoint, kwargs)
    model = kwargs.get("model")

    conn = _generic_cache_conn(cache_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT response_json FROM openai_generic_cache WHERE endpoint=? AND request_hash=? AND url=?",
        (endpoint, req_hash, client_url),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    # Miss: perform call and store
    resp: Any = client.responses.create(**kwargs)  # type: ignore

    resp_json = _response_to_json_str(resp)
    conn.execute(
        "INSERT OR REPLACE INTO openai_generic_cache (endpoint, request_hash, model, url, request_json, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (endpoint, req_hash, model, client_url, req_json, resp_json, time.time()),
    )
    conn.commit()
    conn.close()
    return json.loads(resp_json)


def cached_mistral_chat_complete(
    client: Mistral,
    cache_path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Cached wrapper for Mistral chat.complete API.

    Args:
        client: Mistral client instance  
        cache_path: Path to SQLite cache database
        **kwargs: Arguments passed to chat.complete

    Returns:
        Response in OpenAI-compatible format
    """
    endpoint = "mistral.chat.complete"
    req_hash, req_json = _compute_request_hash(endpoint, kwargs)
    model = kwargs.get("model")
    client_url = "https://api.mistral.ai/"

    conn = _generic_cache_conn(cache_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT response_json FROM openai_generic_cache WHERE endpoint=? AND request_hash=? AND url=?",
        (endpoint, req_hash, client_url),
    )
    row = cur.fetchone()
    if row:
        conn.close()
        return json.loads(row[0])

    # Miss: perform call and store
    response = client.chat.complete(**kwargs)
    resp_json = _response_to_json_str(response)

    conn.execute(
        "INSERT OR REPLACE INTO openai_generic_cache (endpoint, request_hash, model, url, request_json, response_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (endpoint, req_hash, model, client_url, req_json, resp_json, time.time()),
    )
    conn.commit()
    conn.close()
    return json.loads(resp_json)

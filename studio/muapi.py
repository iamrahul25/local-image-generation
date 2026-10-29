"""Python port of the repo's MuAPI client (packages/studio/src/muapi.js)."""
import time
from pathlib import Path
from typing import Callable, Optional

import requests

BASE_URL = "https://api.muapi.ai"
SUCCESS = {"completed", "succeeded", "success"}
FAILURE = {"failed", "error", "cancelled", "canceled"}


class MuapiError(RuntimeError):
    pass


def _headers(api_key: str) -> dict:
    if not api_key:
        raise MuapiError("MuAPI key is missing. Add it in the sidebar (Settings -> API keys).")
    return {"x-api-key": api_key}


def get_balance(api_key: str) -> dict:
    r = requests.get(f"{BASE_URL}/api/v1/account/balance", headers=_headers(api_key), timeout=20)
    if not r.ok:
        raise MuapiError(f"Balance request failed: {r.status_code} {r.text[:200]}")
    return r.json()


def upload_file(api_key: str, filename: str, data: bytes, mime: str = "application/octet-stream") -> str:
    r = requests.post(f"{BASE_URL}/api/v1/upload_file", headers=_headers(api_key),
                      files={"file": (filename, data, mime)}, timeout=300)
    if not r.ok:
        raise MuapiError(f"File upload failed: {r.status_code} {r.text[:200]}")
    body = r.json()
    url = body.get("url") or body.get("file_url") or (body.get("data") or {}).get("url")
    if not url:
        raise MuapiError(f"Upload returned no URL: {body}")
    return url


def submit(api_key: str, endpoint: str, payload: dict) -> dict:
    r = requests.post(f"{BASE_URL}/api/v1/{endpoint}", headers=_headers(api_key), json=payload, timeout=120)
    if not r.ok:
        raise MuapiError(f"API request failed: {r.status_code} {r.reason} - {r.text[:300]}")
    return r.json()


def poll(api_key: str, request_id: str, *, interval: float = 2.0, max_attempts: int = 900,
         on_status: Optional[Callable[[str, int], None]] = None) -> dict:
    url = f"{BASE_URL}/api/v1/predictions/{request_id}/result"
    for attempt in range(1, max_attempts + 1):
        time.sleep(interval)
        try:
            r = requests.get(url, headers=_headers(api_key), timeout=30)
        except requests.RequestException:
            continue
        if not r.ok:
            if r.status_code >= 500:
                continue
            raise MuapiError(f"Poll failed: {r.status_code} {r.text[:200]}")
        result = r.json()
        status = str(result.get("status", "")).lower()
        if on_status:
            on_status(status or "processing", attempt)
        if status in SUCCESS:
            return result
        if status in FAILURE:
            err = result.get("error")
            if isinstance(err, dict):
                err = err.get("message")
            raise MuapiError(f"Generation failed: {err or result.get('message') or 'unknown error'}")
    raise MuapiError(f"Generation timed out. Request ID: {request_id}")


def output_urls(result: dict) -> list[str]:
    urls = []
    for key in ("outputs", "images"):
        val = result.get(key)
        if isinstance(val, list):
            urls += [u if isinstance(u, str) else (u or {}).get("url") for u in val]
    for val in (result.get("url"), (result.get("output") or {}).get("url") if isinstance(result.get("output"), dict) else None):
        if isinstance(val, str):
            urls.append(val)
    return [u for u in dict.fromkeys(urls) if u]


def run(api_key: str, endpoint: str, payload: dict, on_status=None, max_attempts: int = 900) -> dict:
    submitted = submit(api_key, endpoint, payload)
    request_id = submitted.get("request_id") or submitted.get("id")
    if not request_id:
        return {"result": submitted, "urls": output_urls(submitted), "request_id": None}
    result = poll(api_key, request_id, on_status=on_status, max_attempts=max_attempts)
    return {"result": result, "urls": output_urls(result), "request_id": request_id}


def download_output(url: str, dest_dir: Path, prefix: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = url.split("?")[0].rsplit("/", 1)[-1] or "output"
    path = dest_dir / f"{prefix}-{int(time.time())}-{name}"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return path

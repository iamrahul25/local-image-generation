import json
import time
from pathlib import Path

from . import config


def add(kind: str, engine: str, model: str, prompt: str, files: list[str], params: dict | None = None,
        urls: list[str] | None = None) -> dict:
    entry = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "engine": engine, "model": model,
        "prompt": prompt, "files": files, "urls": urls or [], "params": params or {},
    }
    with open(config.HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def load(limit: int = 200) -> list[dict]:
    if not config.HISTORY_FILE.exists():
        return []
    lines = config.HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    out = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def media_kind(path: str) -> str:
    ext = Path(path.split("?")[0]).suffix.lower()
    if ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
        return "image"
    if ext in (".mp4", ".webm", ".mov", ".mkv", ".avi"):
        return "video"
    if ext in (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"):
        return "audio"
    return "file"

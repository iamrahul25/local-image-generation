import os
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import requests

ProgressCb = Optional[Callable[[int, int], None]]


def download_file(url: str, dest: Path, progress: ProgressCb = None, chunk: int = 1 << 20, retries: int = 20) -> Path:
    """Download with resume support via a `.part` file; retries dropped connections."""
    for attempt in range(1, retries + 1):
        try:
            return _download_once(url, Path(dest), progress, chunk)
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries:
                raise
            time.sleep(min(30, 2 * attempt))
    return Path(dest)


def _download_once(url: str, dest: Path, progress: ProgressCb, chunk: int) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        if r.status_code == 416:  # already complete
            part.rename(dest)
            return dest
        r.raise_for_status()
        if have and r.status_code != 206:
            have = 0  # server ignored Range; restart
        total = int(r.headers.get("Content-Length", 0)) + have
        mode = "ab" if have else "wb"
        done = have
        with open(part, mode) as f:
            for block in r.iter_content(chunk_size=chunk):
                if not block:
                    continue
                f.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
    os.replace(part, dest)
    return dest


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)


def human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

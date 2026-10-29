"""Local video/image/audio generation through Wan2GP (the repo's second local engine).

The desktop app talks to a Wan2GP Gradio server; here we call WanGP's in-process
Python API through scripts/wan2gp_worker.py running inside Wan2GP's own venv.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from . import config
from .sysinfo import port_listening

MARKER = "@@WGP@@ "
WEBUI_PORT = 7860

# Curated shortlist shown first in the UI; everything else comes from WanGP's own catalog.
RECOMMENDED = {
    "t2v_1.3B": {"label": "Wan 2.1 T2V 1.3B (default, fits 6 GB)", "fps": 16, "resolutions": ["832x480", "480x832", "624x624"],
                 "steps": 30, "guidance": 5.0, "flow_shift": 5.0, "frames": 81},
    "ti2v_2_2_fastwan": {"label": "Wan 2.2 TI2V 5B FastWan (text/image to video, 3 steps)", "fps": 24,
                         "resolutions": ["832x480", "480x832", "1280x720"], "steps": 3, "guidance": 1.0,
                         "flow_shift": 3.0, "frames": 121},
    "ltxv_distilled": {"label": "LTX Video 13B distilled (fast)", "fps": 30, "resolutions": ["832x480", "480x832", "1280x704"],
                       "steps": 8, "guidance": 1.0, "flow_shift": None, "frames": 97},
}


WAN2GP_REPO = "https://github.com/deepbeepmeep/Wan2GP.git"
TORCH_PACKAGES = ["torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def installed() -> bool:
    return config.WAN2GP_PY.exists() and (config.WAN2GP_DIR / "wgp.py").exists()


def _uv_cmd() -> list[str]:
    if importlib.util.find_spec("uv"):
        return [sys.executable, "-m", "uv"]
    if shutil.which("uv"):
        return [shutil.which("uv")]
    for py in ("python", "py"):
        exe = shutil.which(py)
        if exe and subprocess.run([exe, "-m", "uv", "--version"], capture_output=True,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode == 0:
            return [exe, "-m", "uv"]
    raise RuntimeError("uv was not found. Run setup.bat once, or `pip install uv`.")


def _stream(cmd: list[str], cwd: Path, on_line: Callable[[str], None]) -> None:
    on_line("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, env=_worker_env(),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for line in proc.stdout:
        if line.strip():
            on_line(line.rstrip()[-500:])
    if proc.wait() != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def install(on_line: Callable[[str], None]) -> None:
    """Same steps as setup.bat [5/6]: clone Wan2GP and build its Python 3.11 + PyTorch CUDA venv (~6 GB)."""
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required: https://git-scm.com/download/win")
    uv = _uv_cmd()
    if not (config.WAN2GP_DIR / "wgp.py").exists():
        _stream([git, "clone", "--depth", "1", WAN2GP_REPO, str(config.WAN2GP_DIR)], config.ROOT, on_line)
    if not config.WAN2GP_PY.exists():
        _stream([*uv, "venv", ".venv", "--python", "3.11"], config.WAN2GP_DIR, on_line)
    py = str(config.WAN2GP_PY)
    _stream([*uv, "pip", "install", "--python", py, *TORCH_PACKAGES, "--index-url", TORCH_INDEX], config.WAN2GP_DIR, on_line)
    _stream([*uv, "pip", "install", "--python", py, "-r", "requirements.txt", "--index-strategy", "unsafe-best-match"],
            config.WAN2GP_DIR, on_line)
    if os.name == "nt":
        try:
            _stream([*uv, "pip", "install", "--python", py, "-U", "triton-windows>=3.6,<3.7"], config.WAN2GP_DIR, on_line)
        except RuntimeError as exc:
            on_line(f"triton-windows is optional, continuing without it: {exc}")


def torch_ok() -> tuple[bool, str]:
    if not installed():
        return False, "Wan2GP environment not installed"
    out = subprocess.run([str(config.WAN2GP_PY), "-c",
                          "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                         capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return out.returncode == 0 and "True" in out.stdout, (out.stdout or out.stderr).strip()[-300:]


def _worker_cmd(settings: dict, *extra: str) -> list[str]:
    return [str(config.WAN2GP_PY), "-u", str(config.WAN2GP_WORKER), "--root", str(config.WAN2GP_DIR),
            "--profile", str(settings.get("wan2gp_profile", 4)), "--attention", settings.get("wan2gp_attention", "sdpa"),
            *extra]


def _worker_env() -> dict:
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_XET", "1")  # hf_xet downloads stall on some Windows setups
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _run_worker(cmd: list[str], on_event: Optional[Callable[[dict], None]] = None) -> list[dict]:
    events: list[dict] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(config.WAN2GP_DIR),
                            text=True, encoding="utf-8", errors="replace", bufsize=1, env=_worker_env(),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    raw_tail: list[str] = []
    for line in proc.stdout:
        if line.startswith(MARKER):
            try:
                ev = json.loads(line[len(MARKER):])
            except json.JSONDecodeError:
                continue
            events.append(ev)
            if on_event:
                on_event(ev)
        else:
            raw_tail = (raw_tail + [line.rstrip()])[-40:]
            if on_event and line.strip():
                on_event({"event": "log", "text": line.rstrip()[-500:]})
    proc.wait()
    if proc.returncode != 0 and not any(e["event"] == "error" for e in events):
        events.append({"event": "error", "message": f"Wan2GP worker exited with {proc.returncode}",
                       "traceback": "\n".join(raw_tail)})
    return events


def list_models(settings: dict, refresh: bool = False) -> list[dict]:
    """Cached WanGP catalog. Loading WanGP takes ~30-60s, so only rebuild on explicit refresh."""
    cache = config.WAN2GP_MODELS_CACHE
    if not refresh:
        return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    if not installed():
        return []
    events = _run_worker(_worker_cmd(settings, "list-models", "--out", str(cache)))
    err = next((e for e in events if e["event"] == "error"), None)
    if err:
        raise RuntimeError(err["message"] + "\n" + err.get("traceback", ""))
    return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []


def models_for(models: list[dict], output: str) -> list[dict]:
    def outputs(m):
        mo = m.get("main_output") or []
        return mo if isinstance(mo, list) else [mo]
    return [m for m in models if output in outputs(m)]


def is_available(model: dict) -> bool:
    av = model.get("availability")
    if isinstance(av, dict):
        return bool(av.get("available") or av.get("status") in ("available", "complete", "downloaded"))
    return bool(av)


DOWNLOAD_ERRORS = ("Unable to Download", "Connection broken", "IncompleteRead", "ReadTimeout", "ConnectionError")


def _retry_downloads(run: Callable[[], dict], on_event: Optional[Callable[[dict], None]], retries: int) -> dict:
    """If model weights fail to download, retry (Hugging Face resumes partial files)."""
    for attempt in range(1, retries + 1):
        try:
            return run()
        except RuntimeError as exc:
            if attempt == retries or not any(s in str(exc) for s in DOWNLOAD_ERRORS):
                raise
            if on_event:
                on_event({"event": "status", "text": f"Model download interrupted, resuming (attempt {attempt + 1})..."})
            time.sleep(5)
    raise RuntimeError("unreachable")


def generate(settings: dict, task: dict, on_event: Optional[Callable[[dict], None]] = None,
             download_retries: int = 8) -> dict:
    return _retry_downloads(lambda: _generate_once(settings, task, on_event), on_event, download_retries)


def download_model(settings: dict, model_type: str, on_event: Optional[Callable[[dict], None]] = None,
                   download_retries: int = 8) -> dict:
    """Download all weights for one WanGP model without generating anything."""
    done = _retry_downloads(lambda: _download_once(settings, model_type, on_event), on_event, download_retries)
    cache = config.WAN2GP_MODELS_CACHE
    if cache.exists() and done.get("availability"):
        models = json.loads(cache.read_text(encoding="utf-8"))
        for m in models:
            if m.get("model_type") == model_type:
                m["availability"] = done["availability"]
        cache.write_text(json.dumps(models, indent=1), encoding="utf-8")
    return done


def _download_once(settings: dict, model_type: str, on_event: Optional[Callable[[dict], None]] = None) -> dict:
    if not installed():
        raise RuntimeError("Wan2GP is not installed. See 'Models & Engines'.")
    events = _run_worker(_worker_cmd(settings, "download", model_type), on_event)
    err = next((e for e in events if e["event"] == "error"), None)
    if err:
        raise RuntimeError(err["message"] + "\n" + err.get("traceback", ""))
    done = next((e for e in reversed(events) if e["event"] == "downloaded"), None)
    if not done:
        raise RuntimeError("Wan2GP download finished without a result.")
    return done


def _generate_once(settings: dict, task: dict, on_event: Optional[Callable[[dict], None]] = None) -> dict:
    if not installed():
        raise RuntimeError("Wan2GP is not installed. See 'Models & Engines'.")
    out_dir = config.OUTPUTS_DIR / ("videos" if task.get("_kind", "video") == "video" else task["_kind"] + "s")
    out_dir.mkdir(parents=True, exist_ok=True)
    job = {k: v for k, v in task.items() if not k.startswith("_")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(job, f)
        job_file = f.name
    try:
        events = _run_worker(_worker_cmd(settings, "generate", "--settings", job_file, "--output-dir", str(out_dir)),
                             on_event)
    finally:
        Path(job_file).unlink(missing_ok=True)
    err = next((e for e in events if e["event"] == "error"), None)
    if err:
        raise RuntimeError(err["message"] + "\n" + err.get("traceback", ""))
    done = next((e for e in reversed(events) if e["event"] == "done"), None)
    if not done:
        raise RuntimeError("Wan2GP finished without a result.")
    if not done["success"]:
        raise RuntimeError("Wan2GP generation failed: " + "; ".join(done.get("errors") or ["unknown error"]))
    return done


# ── Optional: full Wan2GP web UI ─────────────────────────────────────────────
def webui_running() -> bool:
    if not port_listening(WEBUI_PORT):
        return False
    try:
        return requests.get(f"http://127.0.0.1:{WEBUI_PORT}/config", timeout=2).ok
    except requests.RequestException:
        return False


def start_webui(settings: dict) -> subprocess.Popen:
    log = open(config.ENGINES_DIR / "wan2gp_webui.log", "a", encoding="utf-8")
    return subprocess.Popen(
        [str(config.WAN2GP_PY), "wgp.py", "--server-port", str(WEBUI_PORT),
         "--attention", settings.get("wan2gp_attention", "sdpa"), "--profile", str(settings.get("wan2gp_profile", 4))],
        cwd=str(config.WAN2GP_DIR), stdout=log, stderr=subprocess.STDOUT, env=_worker_env(),
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def wait_ready(timeout: float = 5) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if webui_running():
            return True
        time.sleep(1)
    return False

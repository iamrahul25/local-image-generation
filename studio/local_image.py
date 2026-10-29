"""Local image generation through stable-diffusion.cpp (the repo's bundled engine).

Mirrors Open-Generative-AI/electron/lib/localInference.js: same model catalog,
same model folder layout and the same sd-cli flags, plus low-VRAM switches.
"""
import json
import random
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from . import config
from .downloads import download_file, extract_zip

SDCPP_RELEASES_API = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=15"
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
STEP_PATTERNS = [
    re.compile(r"step\s+(\d+)\s*/\s*(\d+)", re.I),
    re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*[\d.]+(?:s/it|it/s)", re.I),
]

SAMPLERS = ["euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd"]
SCHEDULERS = ["default", "discrete", "karras", "exponential", "ays", "gits", "simple", "smoothstep", "sgm_uniform"]
ASPECT_RATIOS = ["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"]


# Models not in the repo's catalog. "aux" maps sd-cli roles to files; a None value reuses the repo's file.
EXTRA_MODELS = [
    {
        "id": "flux2-klein-4b",
        "name": "FLUX.2 Klein 4B (Q4) — edit with reference images",
        "description": "Understands reference images: put a garment on a model, change background, restyle a product, "
                       "combine several references. Also good text-to-image. 4 steps, ~1–2 min on 6 GB. Apache-2.0.",
        "type": "flux2",
        "filename": "flux-2-klein-4b-Q4_0.gguf",
        "downloadUrl": "https://huggingface.co/leejet/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf",
        "sizeGB": 2.5,
        "defaultSteps": 4,
        "defaultGuidance": 1.0,
        "sampler": "euler",
        "supportsReference": True,
        "aux": {
            "llm": None,
            "vae": {"displayName": "FLUX.2 VAE", "filename": "flux2-vae.safetensors",
                    "downloadUrl": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"},
        },
    },
]
REFERENCE_MODEL_ID = "flux2-klein-4b"
DIFFUSION_MODEL_TYPES = ("z-image", "flux", "flux2")


def load_catalog() -> dict:
    data = json.loads(config.CATALOG_FILE.read_text(encoding="utf-8"))
    return data["sdcpp"]


def _aux_files(model: dict, cat: dict) -> dict:
    """Role ('llm', 'vae') -> {displayName, filename, downloadUrl} for the files a model needs besides its weights."""
    if "aux" in model:
        return {role: spec or cat["auxiliary"][role] for role, spec in model["aux"].items()}
    return dict(cat["auxiliary"]) if model.get("requiresAuxiliary") else {}


def list_models() -> list[dict]:
    cat = load_catalog()
    out = []
    for m in cat["models"] + EXTRA_MODELS:
        path = config.MODELS_DIR / m["filename"]
        entry = dict(m)
        entry["path"] = str(path)
        entry["downloaded"] = path.exists()
        entry["partial"] = path.with_name(path.name + ".part").exists()
        entry["aux_files"] = {role: str(config.MODELS_DIR / a["filename"]) for role, a in _aux_files(m, cat).items()}
        entry["aux_ready"] = all(Path(p).exists() for p in entry["aux_files"].values())
        entry["ready"] = entry["downloaded"] and entry["aux_ready"]
        out.append(entry)
    return out


def get_model(model_id: str) -> Optional[dict]:
    return next((m for m in list_models() if m["id"] == model_id), None)


def binary_installed() -> bool:
    return config.SD_CLI.exists()


def install_binary(progress: Optional[Callable[[str, float], None]] = None, backend: str = "cuda12") -> Path:
    """Download the latest sd.cpp Windows build (CUDA 12 by default) plus the CUDA runtime DLLs."""
    if binary_installed():
        return config.SD_CLI
    releases = requests.get(SDCPP_RELEASES_API, timeout=30).json()
    for rel in releases:
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        main = next((n for n in assets if f"bin-win-{backend}-x64" in n), None)
        if not main:
            continue
        wanted = [main]
        if backend.startswith("cuda"):
            wanted += [n for n in assets if n.startswith("cudart-") and "win" in n]
        for name in wanted:
            zip_path = config.BIN_DIR / name
            download_file(assets[name], zip_path,
                          lambda d, t, n=name: progress and progress(f"Downloading {n}", d / t if t else 0))
            extract_zip(zip_path, config.BIN_DIR)
            zip_path.unlink(missing_ok=True)
        found = next(config.BIN_DIR.rglob(config.SD_CLI.name), None)
        if found and found != config.SD_CLI:
            for f in found.parent.iterdir():  # flatten nested dirs so DLLs sit beside sd-cli
                target = config.BIN_DIR / f.name
                if not target.exists():
                    f.replace(target)
        if not config.SD_CLI.exists():
            raise RuntimeError(f"Extracted {main} but {config.SD_CLI.name} was not found in {config.BIN_DIR}")
        return config.SD_CLI
    raise RuntimeError(f"No sd.cpp release with a win-{backend}-x64 build found in the last 15 releases.")


def download_model(model_id: str, progress: Optional[Callable[[str, float], None]] = None) -> None:
    cat = load_catalog()
    model = next((m for m in cat["models"] + EXTRA_MODELS if m["id"] == model_id), None)
    if not model:
        raise ValueError(f"Unknown local model: {model_id}")
    jobs = [(model["name"], model["downloadUrl"], model["filename"])]
    jobs += [(a["displayName"], a["downloadUrl"], a["filename"]) for a in _aux_files(model, cat).values()]
    for label, url, filename in jobs:
        download_file(url, config.MODELS_DIR / filename,
                      lambda d, t, lbl=label: progress and progress(f"Downloading {lbl}", d / t if t else 0))


def delete_model(model_id: str) -> None:
    m = get_model(model_id)
    if not m:
        return
    p = Path(m["path"])
    p.unlink(missing_ok=True)
    p.with_name(p.name + ".part").unlink(missing_ok=True)


def ar_to_dimensions(ar: str, base: int) -> tuple[int, int]:
    w_ratio, h_ratio = (int(x) for x in ar.split(":"))
    if w_ratio >= h_ratio:
        h = base
        w = round(base * w_ratio / h_ratio / 64) * 64
    else:
        w = base
        h = round(base * h_ratio / w_ratio / 64) * 64
    return w, h


def build_args(model: dict, prompt: str, out_path: Path, *, negative_prompt: str = "", width: int, height: int,
               steps: int, cfg_scale: float, seed: int, sampler: str, scheduler: str = "default",
               init_image: Optional[Path] = None, strength: float = 0.75, batch_count: int = 1,
               ref_images: Optional[list[Path]] = None,
               low_vram: bool = True, flash_attention: bool = True, extra_args: str = "") -> list[str]:
    model_flag = "--diffusion-model" if model["type"] in DIFFUSION_MODEL_TYPES else "-m"
    args = [
        str(config.SD_CLI), model_flag, model["path"],
        "-p", prompt, "-o", str(out_path),
        "--steps", str(steps), "-W", str(width), "-H", str(height),
        "--cfg-scale", str(cfg_scale), "--seed", str(seed),
        "--sampling-method", sampler, "-b", str(batch_count), "-v",
    ]
    if negative_prompt:
        args += ["-n", negative_prompt]
    for role, path in model.get("aux_files", {}).items():
        args += [f"--{role}", path]
    if scheduler and scheduler != "default":
        args += ["--scheduler", scheduler]
    elif model.get("scheduler"):
        args += ["--scheduler", model["scheduler"]]
    if init_image:
        args += ["-i", str(init_image), "--strength", str(strength)]
    for ref in ref_images or []:
        args += ["-r", str(ref)]
    if low_vram:
        args += ["--offload-to-cpu", "--vae-tiling"]
    if flash_attention:
        args += ["--diffusion-fa"]
    if extra_args.strip():
        args += extra_args.split()
    return args


def generate(model_id: str, prompt: str, *, aspect_ratio: str = "1:1", width: Optional[int] = None,
             height: Optional[int] = None, steps: Optional[int] = None, cfg_scale: Optional[float] = None,
             seed: int = -1, sampler: Optional[str] = None, scheduler: str = "default", negative_prompt: str = "",
             init_image: Optional[Path] = None, strength: float = 0.75, batch_count: int = 1,
             ref_images: Optional[list[Path]] = None,
             low_vram: bool = True, flash_attention: bool = True, extra_args: str = "",
             on_progress: Optional[Callable[[float, str], None]] = None,
             cancel_event: Optional[threading.Event] = None) -> dict:
    if not binary_installed():
        raise RuntimeError("sd.cpp engine is not installed. Open 'Models & Engines' and install it.")
    model = get_model(model_id)
    if not model:
        raise ValueError(f"Unknown local model: {model_id}")
    if not model["ready"]:
        raise RuntimeError(f"'{model['name']}' is not downloaded yet. Download it in 'Models & Engines'.")
    if ref_images and not model.get("supportsReference"):
        raise RuntimeError(f"'{model['name']}' cannot read reference images. Use FLUX.2 Klein 4B for reference edits.")

    base = model.get("defaultWidth", 512) if model["type"] in ("sd1", "sd2") else 1024
    if not (width and height):
        width, height = ar_to_dimensions(aspect_ratio, base)
    steps = int(steps or model.get("defaultSteps", 20))
    cfg_scale = float(cfg_scale if cfg_scale is not None else model.get("defaultGuidance", 7.5))
    sampler = sampler or model.get("sampler", "euler_a")
    seed = seed if seed not in (None, -1) else random.randint(0, 2**31 - 1)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = config.OUTPUTS_DIR / "images" / f"local-{model_id}-{stamp}-{seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = build_args(model, prompt, out_path, negative_prompt=negative_prompt, width=width, height=height,
                      steps=steps, cfg_scale=cfg_scale, seed=seed, sampler=sampler, scheduler=scheduler,
                      init_image=init_image, strength=strength, batch_count=batch_count,
                      ref_images=ref_images, low_vram=low_vram,
                      flash_attention=flash_attention, extra_args=extra_args)

    started = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(config.BIN_DIR),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log: list[str] = []
    tail = ""
    last_step = 0
    while True:
        if cancel_event is not None and cancel_event.is_set():
            proc.kill()
            raise RuntimeError("Generation cancelled")
        chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
        if not chunk:
            if proc.poll() is not None:
                break
            continue
        text = ANSI_RE.sub("", chunk.decode("utf-8", "replace")).replace("\r", "\n")
        log.append(text)
        combined = tail + text
        for pat in STEP_PATTERNS:
            for m in pat.finditer(combined):
                step, total = int(m.group(1)), int(m.group(2))
                if total > 0 and step > last_step and total == steps:
                    last_step = step
                    if on_progress:
                        on_progress(min(1.0, step / total), f"Sampling step {step}/{total}")
        tail = combined[-1024:]
        if last_step == 0 and on_progress:
            on_progress(0.0, f"Loading model... ({int(time.time() - started)}s)")
    proc.wait()
    full_log = "".join(log)
    if proc.returncode != 0:
        lines = [l for l in full_log.splitlines() if l.strip()]
        hint = ""
        if "enough memory" in full_log or "out of memory" in full_log.lower():
            hint = ("Not enough free VRAM. Close other GPU apps (Ollama, Wan2GP, games, browser GPU acceleration), "
                    "keep Low-VRAM mode on, or lower the base size.\n\n")
        raise RuntimeError(f"{hint}sd-cli exited with code {proc.returncode}:\n" + "\n".join(lines[-25:]))

    outputs = [out_path] if batch_count == 1 else sorted(out_path.parent.glob(out_path.stem + "*.png"))
    outputs = [p for p in outputs if p.exists()]
    if not outputs:
        raise RuntimeError("sd.cpp finished but no image was written.\n" + full_log[-2000:])
    return {
        "files": [str(p) for p in outputs], "seed": seed, "width": width, "height": height, "steps": steps,
        "cfg_scale": cfg_scale, "sampler": sampler, "elapsed": round(time.time() - started, 1), "log": full_log,
    }

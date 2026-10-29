import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_DIR = ROOT / "Open-Generative-AI"
DATA_DIR = ROOT / "data"
CATALOG_FILE = DATA_DIR / "catalog.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
WAN2GP_MODELS_CACHE = DATA_DIR / "wan2gp_models.json"
OUTPUTS_DIR = ROOT / "outputs"
HISTORY_FILE = OUTPUTS_DIR / "history.jsonl"

# Same env var the Open-Generative-AI desktop app honours, so both apps can share
# one folder of sd.cpp binaries and model weights.
LOCAL_AI_DIR = Path(os.environ.get("OPEN_GENERATIVE_AI_LOCAL_AI_DIR") or ROOT / "local-ai")
BIN_DIR = LOCAL_AI_DIR / "bin"
MODELS_DIR = LOCAL_AI_DIR / "models"
TMP_DIR = LOCAL_AI_DIR / "tmp"
SD_CLI = BIN_DIR / ("sd-cli.exe" if os.name == "nt" else "sd-cli")

ENGINES_DIR = ROOT / "engines"
WAN2GP_DIR = ENGINES_DIR / "Wan2GP"
WAN2GP_PY = WAN2GP_DIR / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
WAN2GP_WORKER = ROOT / "scripts" / "wan2gp_worker.py"

DEFAULT_LOCAL_IMAGE_MODEL = "z-image-turbo"
DEFAULT_LOCAL_VIDEO_MODEL = "t2v_1.3B"

DEFAULT_SETTINGS = {
    "muapi_key": "",
    "llm_provider": "Ollama (local)",
    "llm_model": "qwen3:4b-instruct",
    "free_vram_before_local": True,
    "llm_base_url": "",
    "llm_api_keys": {},
    "sdcpp_low_vram": True,
    "wan2gp_profile": 4,
    "wan2gp_attention": "sdpa",
    "default_local_video_model": DEFAULT_LOCAL_VIDEO_MODEL,
}

for _d in (DATA_DIR, OUTPUTS_DIR, BIN_DIR, MODELS_DIR, TMP_DIR, ENGINES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            settings.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")

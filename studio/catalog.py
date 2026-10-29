"""Loads the MuAPI model catalog exported from the repo (scripts/export_catalog.mjs)."""
import json
import shutil
import subprocess
from functools import lru_cache

from . import config

CATEGORY_LABELS = {
    "t2i": "Text to Image",
    "i2i": "Image to Image / Edit",
    "t2v": "Text to Video",
    "i2v": "Image to Video",
    "v2v": "Video to Video / Effects",
    "lipsync": "Lip Sync",
    "audio": "Audio / Music / Speech",
    "recast": "Body Swap / Recast",
    "motion_control": "Motion Control",
}

MEDIA_KINDS = ("image", "video", "audio")
_KEY_KIND = {
    "image_url": "image", "images_list": "image", "image_urls": "image", "last_image": "image",
    "video_url": "video", "videos_list": "video", "video_files": "video",
    "audio_url": "audio", "audios_list": "audio", "audio_files": "audio", "audio": "audio",
}
# Fields the UI renders itself (prompt box) or that need special handling.
SKIP_KEYS = {"prompt", "negative_prompt", "request_id", "loras", "lora_list"}


def refresh_catalog() -> str:
    """Re-export the catalog from the cloned repo (requires Node.js)."""
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required to export the catalog from the repo.")
    out = subprocess.run([node, "--no-warnings", str(config.ROOT / "scripts" / "export_catalog.mjs")],
                         capture_output=True, text=True, cwd=str(config.ROOT))
    if out.returncode != 0:
        raise RuntimeError(out.stderr or out.stdout)
    load_catalog.cache_clear()
    return out.stdout


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    if not config.CATALOG_FILE.exists():
        refresh_catalog()
    return json.loads(config.CATALOG_FILE.read_text(encoding="utf-8"))


def models(category: str) -> list[dict]:
    return load_catalog()["muapi"].get(category, [])


def providers(category: str) -> list[str]:
    return sorted({m.get("provider_name") or "Other" for m in models(category)})


def media_fields(model: dict, category: str) -> list[dict]:
    """Return the upload fields a model needs: [{key, kind, multiple, required, label}]."""
    inputs = model.get("inputs") or {}
    required = set(model.get("required") or [])
    fields: dict[str, dict] = {}

    def add(key, kind, multiple, req, label=None, max_items=None):
        if key and key not in fields:
            fields[key] = {"key": key, "kind": kind, "multiple": multiple, "required": req,
                           "label": label or key.replace("_", " ").title(), "max": max_items}

    for key, spec in inputs.items():
        kind = spec.get("field") if spec.get("field") in MEDIA_KINDS else _KEY_KIND.get(key)
        if kind:
            add(key, kind, spec.get("type") == "array", key in required, spec.get("title"),
                spec.get("maxItems") or model.get("maxImages"))

    image_field = model.get("imageField")
    if image_field:
        req = category in ("i2i", "i2v") and not model.get("imageOptional")
        add(image_field, "image", image_field.endswith("list") or image_field.endswith("urls"), req,
            "Reference image(s)" if category == "i2i" else "Start frame", model.get("maxImages"))
    if model.get("lastImageField"):
        add(model["lastImageField"], "image", False, bool(model.get("endImageRequired")), "End frame (optional)")
    if model.get("videoField"):
        add(model["videoField"], "video", model["videoField"].endswith("list"), True, "Source video")
    if model.get("audioField"):
        add(model["audioField"], "audio", model["audioField"].endswith("list"), True, "Audio",
            model.get("maxAudios"))

    if category == "lipsync":
        add("image_url" if model.get("category") == "image" else "video_url",
            "image" if model.get("category") == "image" else "video", False, True,
            "Portrait image" if model.get("category") == "image" else "Source video")
        add("audio_url", "audio", False, True, "Driving audio")
    if category in ("v2v", "recast", "motion_control") and not any(f["kind"] == "video" for f in fields.values()):
        add("video_url", "video", False, True, "Source video")
    if category == "motion_control":
        add("images_list", "image", True, True, "Reference images")
    return list(fields.values())


def param_fields(model: dict) -> dict:
    media_keys = {f["key"] for f in media_fields(model, "")}
    return {k: v for k, v in (model.get("inputs") or {}).items()
            if k not in SKIP_KEYS and k not in media_keys and k not in (model.get("fixedParameters") or {})}


def has_prompt(model: dict) -> bool:
    return "prompt" in (model.get("inputs") or {}) or bool(model.get("hasPrompt"))


def build_payload(model: dict, prompt: str, negative_prompt: str, params: dict, media: dict) -> dict:
    payload = {}
    payload.update(model.get("fixedParameters") or {})
    if prompt and has_prompt(model):
        payload["prompt"] = prompt
    if negative_prompt and "negative_prompt" in (model.get("inputs") or {}):
        payload["negative_prompt"] = negative_prompt
    for k, v in params.items():
        if v is None or v == "":
            continue
        if k == "seed" and v == -1:
            continue
        payload[k] = v
    payload.update({k: v for k, v in media.items() if v})
    return payload

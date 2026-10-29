"""Reusable Streamlit widgets."""
import json
import mimetypes
from pathlib import Path

import streamlit as st

from . import catalog, history, llm, muapi


# ── Settings access ──────────────────────────────────────────────────────────
def settings() -> dict:
    return st.session_state["settings"]


def llm_call_kwargs() -> dict:
    s = settings()
    provider = s["llm_provider"]
    return {"provider": provider, "model": s["llm_model"], "api_key": s["llm_api_keys"].get(provider, ""),
            "base_url": s.get("llm_base_url", "")}


# ── Prompt box with LLM helpers ──────────────────────────────────────────────
def _enhance_cb(key: str, target: str, target_model: str | None) -> None:
    text = st.session_state.get(key, "").strip()
    if not text:
        st.session_state[f"{key}__err"] = "Type an idea first, then enhance it."
        return
    kw = llm_call_kwargs()
    try:
        st.session_state[f"{key}__orig"] = text
        st.session_state[key] = llm.enhance(kw["provider"], kw["model"], text, target, api_key=kw["api_key"],
                                            base_url=kw["base_url"], target_model=target_model)
        st.session_state.pop(f"{key}__err", None)
    except Exception as exc:  # show error inline, keep the prompt
        st.session_state[f"{key}__err"] = f"LLM error ({kw['provider']} / {kw['model']}): {exc}"


def _negative_cb(key: str, neg_key: str) -> None:
    kw = llm_call_kwargs()
    try:
        st.session_state[neg_key] = llm.enhance(kw["provider"], kw["model"], st.session_state.get(key, ""), "negative",
                                                api_key=kw["api_key"], base_url=kw["base_url"])
    except Exception as exc:
        st.session_state[f"{key}__err"] = f"LLM error: {exc}"


def _undo_cb(key: str) -> None:
    if f"{key}__orig" in st.session_state:
        st.session_state[key] = st.session_state.pop(f"{key}__orig")


def prompt_box(key: str, target: str, *, target_model: str | None = None, negative_key: str | None = None,
               label: str = "Prompt", height: int = 110, placeholder: str = "Describe what you want to create...") -> tuple[str, str]:
    st.session_state.setdefault(key, "")
    st.text_area(label, key=key, height=height, placeholder=placeholder)
    s = settings()
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.8, 2.6])
    c1.button("✨ Enhance with LLM", key=f"{key}__enh", on_click=_enhance_cb, args=(key, target, target_model),
              width="stretch", help="Rewrite the prompt with the LLM chosen in the sidebar")
    if negative_key:
        c2.button("🚫 Suggest negative", key=f"{key}__neg", on_click=_negative_cb, args=(key, negative_key),
                  width="stretch")
    if f"{key}__orig" in st.session_state:
        c3.button("↩ Undo", key=f"{key}__undo", on_click=_undo_cb, args=(key,), width="stretch")
    c4.caption(f"LLM: **{s['llm_provider']}** · `{s['llm_model']}`")
    if st.session_state.get(f"{key}__err"):
        st.warning(st.session_state.pop(f"{key}__err"))
    negative = ""
    if negative_key:
        st.session_state.setdefault(negative_key, "")
        negative = st.text_input("Negative prompt", key=negative_key,
                                 placeholder="blurry, low quality, watermark, extra fingers...")
    return st.session_state[key], negative


# ── Schema-driven parameter form (MuAPI models) ──────────────────────────────
def _num(spec, *names):
    for n in names:
        if spec.get(n) is not None:
            return spec[n]
    return None


def _clamp(val, lo, hi):
    if lo is not None and val < lo:
        return type(val)(lo)
    if hi is not None and val > hi:
        return type(val)(hi)
    return val


def render_params(fields: dict, key_prefix: str, columns: int = 3) -> dict:
    values = {}
    items = list(fields.items())
    if not items:
        return values
    cols = st.columns(columns)
    for i, (name, spec) in enumerate(items):
        col = cols[i % columns]
        t = spec.get("type")
        title = spec.get("title") or name.replace("_", " ").title()
        help_ = spec.get("description")
        default = spec.get("default")
        k = f"{key_prefix}__{name}"
        with col:
            if "enum" in spec:
                options = list(spec["enum"])
                idx = options.index(default) if default in options else 0
                values[name] = st.selectbox(title, options, index=idx, key=k, help=help_, format_func=str)
            elif t == "boolean":
                values[name] = st.checkbox(title, value=bool(default), key=k, help=help_)
            elif t in ("int", "integer"):
                lo, hi = _num(spec, "minValue", "minimum"), _num(spec, "maxValue", "maximum")
                if name == "seed":
                    values[name] = int(st.number_input(title, value=int(default if default is not None else -1),
                                                       min_value=-1, step=1, key=k, help="-1 = random"))
                else:
                    lo = None if lo is None else int(lo)
                    hi = None if hi is None else int(hi)
                    val = _clamp(int(default if default is not None else (lo if lo is not None else 0)), lo, hi)
                    values[name] = int(st.number_input(title, value=val, min_value=lo, max_value=hi,
                                                       step=max(1, int(spec.get("step") or 1)), key=k, help=help_))
            elif t in ("number", "float"):
                lo, hi = _num(spec, "minValue", "minimum"), _num(spec, "maxValue", "maximum")
                val = _clamp(float(default if default is not None else (lo if lo is not None else 0.0)), lo, hi)
                values[name] = float(st.number_input(title, value=val, min_value=None if lo is None else float(lo),
                                                     max_value=None if hi is None else float(hi),
                                                     step=float(spec.get("step") or 0.05), key=k, help=help_))
            elif t == "array":
                raw = st.text_input(f"{title} (JSON list or comma-separated)", key=k, help=help_,
                                    value=json.dumps(default) if default else "")
                values[name] = _parse_list(raw)
            else:
                if spec.get("format") == "text" or name in ("lyrics", "text", "style"):
                    values[name] = st.text_area(title, value=default or "", key=k, help=help_,
                                                placeholder=spec.get("placeholder") or "", height=80)
                else:
                    values[name] = st.text_input(title, value=str(default) if default is not None else "", key=k,
                                                 help=help_, placeholder=spec.get("placeholder") or "")
    return values


def _parse_list(raw: str):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        val = json.loads(raw)
        return val if isinstance(val, list) else [val]
    except json.JSONDecodeError:
        return [x.strip() for x in raw.split(",") if x.strip()]


# ── Media inputs for cloud models ────────────────────────────────────────────
_EXT = {"image": ["png", "jpg", "jpeg", "webp"], "video": ["mp4", "mov", "webm"], "audio": ["mp3", "wav", "m4a", "ogg", "flac"]}


def media_inputs(fields: list[dict], key_prefix: str) -> list[tuple[dict, list, str]]:
    """Render upload + URL widgets. Returns [(field, uploaded_files, pasted_urls)]."""
    out = []
    for f in fields:
        label = f"{f['label']}{' *' if f['required'] else ''}"
        if f["multiple"] and f.get("max"):
            label += f" (up to {f['max']})"
        c1, c2 = st.columns([3, 2])
        files = c1.file_uploader(label, type=_EXT[f["kind"]], accept_multiple_files=f["multiple"],
                                 key=f"{key_prefix}__up__{f['key']}")
        urls = c2.text_input("…or paste URL(s)", key=f"{key_prefix}__url__{f['key']}",
                             placeholder="https://... (comma-separated)")
        files = files if isinstance(files, list) else ([files] if files else [])
        for up in files:
            if f["kind"] == "image":
                c1.image(up, width=160)
        out.append((f, files, urls))
    return out


def resolve_media(api_key: str, rendered: list[tuple[dict, list, str]]) -> dict:
    media = {}
    for f, files, urls in rendered:
        values = [u.strip() for u in (urls or "").split(",") if u.strip()]
        for up in files:
            mime = up.type or mimetypes.guess_type(up.name)[0] or "application/octet-stream"
            values.append(muapi.upload_file(api_key, up.name, up.getvalue(), mime))
        if f["required"] and not values:
            raise ValueError(f"'{f['label']}' is required for this model.")
        if values:
            if f.get("max"):
                values = values[: int(f["max"])]
            media[f["key"]] = values if f["multiple"] else values[0]
    return media


# ── Output rendering ─────────────────────────────────────────────────────────
def show_outputs(files: list[str], urls: list[str] | None = None, fallback_kind: str = "image",
                 key_prefix: str = "out") -> None:
    items = list(files) or list(urls or [])
    if not items:
        return
    cols = st.columns(min(len(items), 3))
    for i, item in enumerate(items):
        kind = history.media_kind(item)
        if kind == "file":
            kind = fallback_kind
        with cols[i % len(cols)]:
            is_local = Path(item).exists()
            if kind == "image":
                st.image(item, width="stretch")
            elif kind == "video":
                st.video(item)
            elif kind == "audio":
                st.audio(item)
            if is_local:
                p = Path(item)
                st.download_button("⬇ Download", p.read_bytes(), file_name=p.name, key=f"{key_prefix}_dl_{p.name}_{i}",
                                   width="stretch")
                st.caption(f"`{p}`")
            else:
                st.markdown(f"[Open original]({item})")


def cloud_model_picker(categories: list[str], key: str) -> tuple[str, dict]:
    c1, c2, c3 = st.columns([1.3, 1.2, 2.5])
    category = c1.selectbox("Mode", categories, format_func=lambda c: catalog.CATEGORY_LABELS.get(c, c),
                            key=f"{key}__cat")
    provs = ["All providers"] + catalog.providers(category)
    prov = c2.selectbox("Provider", provs, key=f"{key}__prov")
    search = c3.text_input("Search models", key=f"{key}__search", placeholder="flux, kling, veo, suno...")
    models = catalog.models(category)
    if prov != "All providers":
        models = [m for m in models if (m.get("provider_name") or "Other") == prov]
    if search:
        s = search.lower()
        models = [m for m in models if s in m["name"].lower() or s in m["id"].lower()]
    if not models:
        st.info("No models match your filter.")
        st.stop()
    model = st.selectbox(f"Model ({len(models)} available)", models, key=f"{key}__model",
                         format_func=lambda m: f"{m['name']}  ·  {m.get('provider_name') or ''}  ·  {m['id']}")
    if model.get("description"):
        st.caption(model["description"][:400])
    return category, model

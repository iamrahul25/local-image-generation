"""Open Generative AI - Streamlit Studio.

Run:  .venv\\Scripts\\streamlit run app.py
"""
import tempfile
import time
from pathlib import Path

import streamlit as st

from studio import catalog, config, history, llm, local_image, local_video, muapi, sysinfo, ui

st.set_page_config(page_title="Open Generative AI Studio", page_icon="🎨", layout="wide")

if "settings" not in st.session_state:
    st.session_state["settings"] = config.load_settings()
S = st.session_state["settings"]


def persist() -> None:
    config.save_settings(S)


if "flash" in st.session_state:
    st.toast(st.session_state.pop("flash"))


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar: system, API keys, LLM
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🎨 Open Gen AI Studio")
    st.caption("Streamlit front-end for [Open-Generative-AI](https://github.com/Anil-matcha/Open-Generative-AI)")

    snap = sysinfo.snapshot()
    g = snap["gpu"]
    with st.expander("🖥 System", expanded=False):
        st.markdown(f"**CPU:** {snap['cpu']} ({snap['cores']}C/{snap['threads']}T)")
        st.markdown(f"**RAM:** {snap['ram_used_gb']} / {snap['ram_total_gb']} GB")
        if g:
            st.markdown(f"**GPU:** {g['name']}")
            st.progress(g["vram_used_mb"] / g["vram_total_mb"],
                        text=f"VRAM {g['vram_used_mb']} / {g['vram_total_mb']} MB · {g['util']}% · {g['temp']}°C")
        else:
            st.warning("No NVIDIA GPU detected (nvidia-smi failed).")
        st.markdown(f"**Disk free:** {snap['disk_free_gb']} GB")
        st.button("Refresh", key="sys_refresh")

    st.subheader("Engines")
    ok = lambda b: "🟢" if b else "🔴"  # noqa: E731
    img_ready = any(m["ready"] for m in local_image.list_models())
    ollama_up = llm.ollama_running()
    st.markdown(
        f"{ok(local_image.binary_installed())} sd.cpp engine  \n"
        f"{ok(img_ready)} Local image model  \n"
        f"{ok(local_video.installed())} Wan2GP (local video)  \n"
        f"{ok(ollama_up)} Ollama (local LLM)  \n"
        f"{ok(bool(S['muapi_key']))} MuAPI key (cloud)"
    )

    st.subheader("🔑 Cloud API (MuAPI)")
    key = st.text_input("MuAPI key", value=S["muapi_key"], type="password",
                        help="Needed only for cloud models. Get one at muapi.ai")
    if key != S["muapi_key"]:
        S["muapi_key"] = key
        persist()
    if S["muapi_key"] and st.button("Check balance"):
        try:
            st.json(muapi.get_balance(S["muapi_key"]))
        except Exception as exc:
            st.error(str(exc))

    st.subheader("🧠 Prompt LLM")
    providers = list(llm.PROVIDERS)
    provider = st.selectbox("Provider", providers, index=providers.index(S["llm_provider"])
                            if S["llm_provider"] in providers else 0)
    kind, default_base, suggestions = llm.PROVIDERS[provider]
    if provider == "Ollama (local)":
        installed_models = llm.ollama_models()
        options = list(dict.fromkeys(installed_models + suggestions))
        model_name = st.selectbox("Model", options, index=options.index(S["llm_model"]) if S["llm_model"] in options else 0,
                                  help="Models not yet pulled can be downloaded in 'Models & Engines'.")
        if not ollama_up:
            st.caption("⚠ Ollama is not running. Start the Ollama app.")
    else:
        model_name = st.text_input("Model", value=S["llm_model"] if S["llm_provider"] == provider else suggestions[0],
                                   help="Suggestions: " + ", ".join(suggestions))
    base_url = st.text_input("Base URL (optional override)", value=S.get("llm_base_url", "") if S["llm_provider"] == provider else "",
                             placeholder=default_base or "https://your-endpoint/v1")
    if provider not in llm.LOCAL_PROVIDERS:
        api_key = st.text_input(f"{provider} API key", value=S["llm_api_keys"].get(provider, ""), type="password")
        if api_key != S["llm_api_keys"].get(provider, ""):
            S["llm_api_keys"][provider] = api_key
    S.update(llm_provider=provider, llm_model=model_name, llm_base_url=base_url)
    persist()


# ═════════════════════════════════════════════════════════════════════════════
# Shared runners
# ═════════════════════════════════════════════════════════════════════════════
def run_cloud(category: str, model: dict, prompt: str, negative: str, params: dict, rendered_media, kind: str) -> None:
    if not S["muapi_key"]:
        st.error("Add your MuAPI key in the sidebar to use cloud models.")
        return
    if model.get("promptRequired") and not prompt.strip():
        st.error("This model requires a prompt.")
        return
    status = st.status(f"Running {model['name']} on MuAPI...", expanded=True)
    try:
        media = ui.resolve_media(S["muapi_key"], rendered_media)
        payload = catalog.build_payload(model, prompt, negative, params, media)
        status.write("Payload:")
        status.json(payload, expanded=False)
        endpoint = model.get("endpoint") or model["id"]
        res = muapi.run(S["muapi_key"], endpoint, payload,
                        on_status=lambda s, n: status.update(label=f"{model['name']}: {s} ({n * 2}s)"))
        files = []
        for url in res["urls"]:
            try:
                files.append(str(muapi.download_output(url, config.OUTPUTS_DIR / f"{kind}s", f"cloud-{model['id']}")))
            except Exception as exc:
                status.write(f"Could not download {url}: {exc}")
        history.add(kind, "muapi", model["id"], prompt, files, payload, res["urls"])
        status.update(label=f"Done: {model['name']}", state="complete")
        st.session_state[f"last_{kind}"] = {"files": files, "urls": res["urls"]}
        if not res["urls"]:
            st.json(res["result"])
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc))


def free_vram() -> None:
    if S.get("free_vram_before_local", True):
        unloaded = llm.ollama_unload_all()
        if unloaded:
            st.toast(f"Unloaded {', '.join(unloaded)} from VRAM to make room")


def wan_event_logger(status):
    bar = status.progress(0.0)
    log_box = status.empty()
    logs: list[str] = []

    def on_event(ev):
        e = ev["event"]
        if e == "progress":
            pct = float(ev.get("progress") or 0) / 100
            step = f" step {ev['step']}/{ev['total']}" if ev.get("step") and ev.get("total") else ""
            bar.progress(min(max(pct, 0.0), 1.0), text=f"{ev.get('phase') or ev.get('status') or 'Working'}{step}")
        elif e in ("status", "log"):
            text = ev.get("text", "")
            head = text.split("|")[0].split(":")[0]
            if logs and "%|" in text and logs[-1].split(":")[0] == head:
                logs[-1] = text  # collapse tqdm progress-bar refreshes
            else:
                logs.append(text)
            log_box.code("\n".join(logs[-12:]), language=None)

    return on_event


def run_wan2gp(task: dict, kind: str, label: str) -> None:
    free_vram()
    status = st.status(f"{label}: starting Wan2GP (first run downloads the model, can take a while)...", expanded=True)
    on_event = wan_event_logger(status)
    try:
        done = local_video.generate(S, {**task, "_kind": kind}, on_event)
        files = [str(Path(f)) for f in done["files"]]
        history.add(kind, "wan2gp", task["model_type"], task.get("prompt", ""), files, task)
        status.update(label=f"Done in {done.get('elapsed')}s: {label}", state="complete", expanded=False)
        st.session_state[f"last_{kind}"] = {"files": files, "urls": []}
    except Exception as exc:
        status.update(label="Failed", state="error")
        st.error(str(exc)[-3000:])


def wan_available() -> dict[str, bool]:
    try:
        return {x["model_type"]: local_video.is_available(x) for x in local_video.list_models(S)}
    except Exception:
        return {}


def wan_download_button(model_type: str, label: str, key: str) -> None:
    """One-click download of a Wan2GP model's weights (otherwise fetched on first generation)."""
    if not st.button(f"⬇ Download {label}", key=key):
        return
    status = st.status(f"Downloading {label} (several GB, resumes if interrupted)...", expanded=True)
    try:
        done = local_video.download_model(S, model_type, wan_event_logger(status))
        status.update(label=f"Downloaded {label} in {done.get('elapsed')}s", state="complete", expanded=False)
        st.session_state["flash"] = f"✅ {label} is ready"
        st.rerun()
    except Exception as exc:
        status.update(label="Download failed", state="error")
        st.error(str(exc)[-3000:])


def sd_download_button(m: dict, key: str) -> None:
    if not st.button(f"⬇ Download {m['name']} ({m['sizeGB']} GB)", key=key, type="primary"):
        return
    bar = st.progress(0.0)
    try:
        local_image.download_model(m["id"], lambda t, p: bar.progress(min(p, 1.0), text=t))
        st.session_state["flash"] = f"✅ {m['name']} is ready"
        st.rerun()
    except Exception as exc:
        st.error(str(exc))


def sd_install_button(key: str) -> None:
    backend = st.selectbox("sd.cpp build", ["cuda12", "vulkan", "cpu"], key=f"{key}_backend")
    if st.button("⬇ Install sd.cpp engine", key=key, type="primary"):
        bar = st.progress(0.0)
        try:
            local_image.install_binary(lambda t, p: bar.progress(min(p, 1.0), text=t), backend=backend)
            st.session_state["flash"] = "✅ sd.cpp installed"
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def wan_install_button(key: str) -> None:
    st.caption("Clones Wan2GP and installs Python 3.11 + PyTorch CUDA into `engines/Wan2GP` (~6 GB, 5-20 min). "
               "Needs Git installed.")
    if not st.button("⬇ Install Wan2GP", key=key, type="primary"):
        return
    status = st.status("Installing Wan2GP...", expanded=True)
    log_box = status.empty()
    logs: list[str] = []

    def on_line(line: str) -> None:
        logs.append(line)
        log_box.code("\n".join(logs[-15:]), language=None)

    try:
        local_video.install(on_line)
        status.update(label="Wan2GP installed", state="complete", expanded=False)
        st.session_state["flash"] = "✅ Wan2GP installed. Click 'Refresh Wan2GP model list' to see its models."
        st.rerun()
    except Exception as exc:
        status.update(label="Install failed", state="error")
        st.error(str(exc)[-3000:])


def save_upload(up) -> Path:
    tmp = config.TMP_DIR / f"upload-{int(time.time())}-{up.name}"
    tmp.write_bytes(up.getvalue())
    return tmp


def wan_models(output: str) -> list[dict]:
    try:
        return local_video.models_for(local_video.list_models(S), output)
    except Exception as exc:
        st.caption(f"Could not read Wan2GP model list: {exc}")
        return []


# ═════════════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab_img, tab_vid, tab_aud, tab_lip, tab_lab, tab_models, tab_hist = st.tabs(
    ["🖼 Image", "🎬 Video", "🎵 Audio", "🗣 Lip Sync & More", "✍️ Prompt Lab", "📦 Models & Engines", "🕘 History"])

# ── IMAGE ────────────────────────────────────────────────────────────────────
with tab_img:
    engine = st.radio("Engine", ["Local · sd.cpp", "Local · Wan2GP image models", "Cloud · MuAPI (150+ models)"],
                      horizontal=True, key="img_engine")

    if engine == "Local · sd.cpp":
        models = {x["id"]: x for x in local_image.list_models()}
        if st.session_state.get("sd_model") not in models:
            st.session_state["sd_model"] = config.DEFAULT_LOCAL_IMAGE_MODEL
        MODE_T2I, MODE_REF, MODE_I2I = ("Text → Image", "Reference image → New image (understands the image)",
                                        "Image → Image (redraw / restyle the same picture)")
        mode = st.radio("Mode", [MODE_T2I, MODE_REF, MODE_I2I], horizontal=True, key="sd_mode")
        mid = st.selectbox("Local model", list(models), key="sd_model",
                           format_func=lambda k: f"{'✅' if models[k]['ready'] else '⬇'} {models[k]['name']} "
                                                 f"({models[k]['sizeGB']} GB) · {models[k]['type']}")
        m = models[mid]
        st.caption(m["description"])
        if not local_image.binary_installed():
            st.warning("The sd.cpp engine is not installed yet.")
            sd_install_button("sd_install_inline")
        if not m["ready"]:
            st.warning("This model is not downloaded yet.")
            sd_download_button(m, f"sd_dl_inline_{m['id']}")

        init_up, strength, ref_ups = None, 0.75, []
        if mode == MODE_REF:
            if not m.get("supportsReference"):
                ref_name = models[local_image.REFERENCE_MODEL_ID]["name"]
                st.warning(f"**{m['name']}** is a text-only model and cannot read images. "
                           f"Reference edits need an edit model such as **{ref_name}**.")
                st.button(f"Switch to {ref_name}", key="sd_use_ref_model",
                          on_click=lambda: st.session_state.update(sd_model=local_image.REFERENCE_MODEL_ID))
            ref_ups = st.file_uploader("Reference image(s) — e.g. a garment, product, face, or scene (up to 4)",
                                       type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True, key="sd_refs")
            if ref_ups:
                st.image([r.getvalue() for r in ref_ups[:4]], width=140)
            st.caption("Describe the **new** image and refer to the reference, e.g. *“A young female fashion model "
                       "wearing the dress from the reference image, full-body studio photo, soft light, plain grey "
                       "background.”* With several references, say which is which (image 1, image 2...).")
        elif mode == MODE_I2I:
            c1, c2 = st.columns([2, 1])
            init_up = c1.file_uploader("Init image", type=["png", "jpg", "jpeg", "webp"], key="sd_init")
            strength = c2.slider("Strength (how much to change)", 0.05, 1.0, 0.6, 0.05, key="sd_strength")
            st.caption("Classic img2img: adds noise to this picture and redraws it. Low strength keeps the picture; "
                       "high strength follows the prompt but loses the original. It does **not** understand the "
                       "image — to place an item into a new scene (e.g. put clothes on a model) use "
                       "**Reference image → New image**.")
        prompt, negative = ui.prompt_box("img_prompt", "edit" if mode == MODE_REF else "image",
                                         target_model=m["name"], negative_key="img_negative")

        c1, c2, c3, c4 = st.columns(4)
        ar = c1.selectbox("Aspect ratio", local_image.ASPECT_RATIOS, key="sd_ar")
        base_default = 1024 if m["type"] in ("z-image", "sdxl", "flux2") else 512
        base_opts = [512, 640, 768, 896, 1024, 1280]
        base = c2.selectbox("Base size (px)", base_opts, index=base_opts.index(base_default), key=f"sd_base_{m['type']}")
        steps = c3.number_input("Steps", 1, 150, int(m.get("defaultSteps", 20)), key=f"sd_steps_{m['id']}")
        cfg = c4.number_input("CFG scale", 0.0, 30.0, float(m.get("defaultGuidance", 7.5)), 0.5, key=f"sd_cfg_{m['id']}")
        c1, c2, c3, c4 = st.columns(4)
        sampler = c1.selectbox("Sampler", local_image.SAMPLERS,
                               index=local_image.SAMPLERS.index(m.get("sampler", "euler_a"))
                               if m.get("sampler", "euler_a") in local_image.SAMPLERS else 0, key=f"sd_sampler_{m['id']}")
        sched_default = m.get("scheduler", "default")
        scheduler = c2.selectbox("Scheduler", local_image.SCHEDULERS,
                                 index=local_image.SCHEDULERS.index(sched_default) if sched_default in local_image.SCHEDULERS else 0,
                                 key=f"sd_sched_{m['id']}")
        seed = c3.number_input("Seed (-1 = random)", -1, 2**31 - 1, -1, key="sd_seed")
        batch = c4.number_input("Images", 1, 8, 1, key="sd_batch")
        w, h = local_image.ar_to_dimensions(ar, base)
        with st.expander("⚙ Performance / advanced"):
            low_vram = st.checkbox("Low-VRAM mode (offload weights to RAM + VAE tiling) — recommended for 6 GB GPUs",
                                   value=S["sdcpp_low_vram"], key="sd_lowvram")
            fa = st.checkbox("Flash attention in diffusion model", value=True, key="sd_fa")
            extra = st.text_input("Extra sd-cli arguments", key="sd_extra", placeholder="e.g. --clip-skip 2")
            st.caption(f"Output size: **{w}×{h}**")

        if st.button("🎨 Generate image", type="primary", disabled=not m["ready"], key="sd_go"):
            if not prompt.strip():
                st.error("Enter a prompt.")
            elif mode == MODE_REF and not (ref_ups and m.get("supportsReference")):
                st.error("Upload at least one reference image and use a model that supports reference images.")
            elif mode == MODE_I2I and not init_up:
                st.error("Upload an init image.")
            else:
                free_vram()
                bar = st.progress(0.0, text="Starting sd.cpp...")
                try:
                    res = local_image.generate(
                        m["id"], prompt, width=w, height=h, steps=int(steps), cfg_scale=float(cfg), seed=int(seed),
                        sampler=sampler, scheduler=scheduler, negative_prompt=negative,
                        init_image=save_upload(init_up) if init_up else None, strength=strength,
                        ref_images=[save_upload(r) for r in ref_ups[:4]] or None,
                        batch_count=int(batch), low_vram=low_vram, flash_attention=fa, extra_args=extra,
                        on_progress=lambda p, t: bar.progress(p, text=t))
                    bar.progress(1.0, text=f"Done in {res['elapsed']}s · seed {res['seed']}")
                    history.add("image", "sd.cpp", m["id"], prompt, res["files"],
                                {k: v for k, v in res.items() if k not in ("files", "log")})
                    st.session_state["last_image"] = {"files": res["files"], "urls": []}
                except Exception as exc:
                    bar.empty()
                    st.error(str(exc))

    elif engine == "Local · Wan2GP image models":
        if not local_video.installed():
            st.warning("Wan2GP is not installed.")
            wan_install_button("wgp_install_img")
        else:
            wm = wan_models("image")
            if not wm:
                st.info("No Wan2GP image models listed yet. Click 'Refresh Wan2GP model list' in 📦 Models & Engines.")
            else:
                pref = next((i for i, x in enumerate(wm) if x["model_type"] == "z_image"), 0)
                sel = st.selectbox("Wan2GP image model", wm, index=pref, key="wimg_model",
                                   format_func=lambda x: f"{'✅' if local_video.is_available(x) else '⬇'} {x['name']} · {x['model_type']}")
                st.caption("⬇ = not downloaded yet (download below, or it downloads on first use). 6 GB GPU: prefer "
                           "Z-Image and Flux 2 Klein 4B; 12B+ models (Kontext, Qwen Image Edit) are very slow or may not fit.")
                if not local_video.is_available(sel):
                    wan_download_button(sel["model_type"], sel["name"], f"wimg_dl_{sel['model_type']}")
                can_ref = (sel.get("capabilities") or {}).get("image_to_image")
                wref_ups = []
                if can_ref:
                    wref_ups = st.file_uploader("Reference image(s) — this model can read them (garment, product, face...)",
                                                type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True,
                                                key=f"wimg_refs_{sel['model_type']}")
                prompt, negative = ui.prompt_box("wimg_prompt", "edit" if wref_ups else "image",
                                                 target_model=sel["name"], negative_key="wimg_negative")
                c1, c2, c3 = st.columns(3)
                res_ = c1.selectbox("Resolution", ["1024x1024", "1280x720", "720x1280", "1024x768", "768x1024", "832x480"], key="wimg_res")
                steps = c2.number_input("Steps (0 = model default)", 0, 100, 0, key="wimg_steps")
                seed = c3.number_input("Seed (-1 = random)", -1, 2**31 - 1, -1, key="wimg_seed")
                if st.button("🎨 Generate with Wan2GP", type="primary", key="wimg_go"):
                    task = {"model_type": sel["model_type"], "prompt": prompt, "negative_prompt": negative or None,
                            "resolution": res_, "seed": int(seed), "batch_size": 1}
                    if steps:
                        task["num_inference_steps"] = int(steps)
                    if wref_ups:
                        task["image_refs"] = [str(save_upload(r)) for r in wref_ups[:4]]
                    run_wan2gp(task, "image", sel["name"])

    else:
        category, model = ui.cloud_model_picker(["t2i", "i2i"], "cimg")
        prompt, negative = ui.prompt_box("cimg_prompt", "edit" if category == "i2i" else "image", target_model=model["name"],
                                         negative_key="cimg_negative" if "negative_prompt" in (model.get("inputs") or {}) else None)
        rendered = ui.media_inputs(catalog.media_fields(model, category), f"cimg_{model['id']}")
        with st.expander("⚙ Model parameters", expanded=True):
            params = ui.render_params(catalog.param_fields(model), f"cimg_{model['id']}")
        if st.button("☁ Generate on MuAPI", type="primary", key="cimg_go"):
            run_cloud(category, model, prompt, negative, params, rendered, "image")

    if st.session_state.get("last_image"):
        st.divider()
        st.subheader("Result")
        ui.show_outputs(st.session_state["last_image"]["files"], st.session_state["last_image"]["urls"], "image", "res_img")

# ── VIDEO ────────────────────────────────────────────────────────────────────
with tab_vid:
    engine = st.radio("Engine", ["Local · Wan2GP", "Cloud · MuAPI (340+ models)"], horizontal=True, key="vid_engine")

    if engine == "Local · Wan2GP":
        if not local_video.installed():
            st.warning("Wan2GP is not installed yet.")
            wan_install_button("wgp_install_vid")
        else:
            rec_ids = list(local_video.RECOMMENDED)
            others = [x for x in wan_models("video") if x["model_type"] not in rec_ids]
            choices = rec_ids + [x["model_type"] for x in others]
            names = {**{k: v["label"] for k, v in local_video.RECOMMENDED.items()},
                     **{x["model_type"]: f"{x['name']} · {x['model_type']}" for x in others}}
            default_vm = S.get("default_local_video_model", config.DEFAULT_LOCAL_VIDEO_MODEL)
            mt = st.selectbox(f"Local video model ({len(choices)} available in Wan2GP)", choices,
                              index=choices.index(default_vm) if default_vm in choices else 0,
                              format_func=lambda k: names.get(k, k), key="vid_model")
            preset = local_video.RECOMMENDED.get(mt, {"fps": 16, "resolutions": ["832x480", "480x832", "1280x720"],
                                                      "steps": 0, "guidance": None, "flow_shift": None, "frames": 81})
            if mt not in local_video.RECOMMENDED:
                st.caption("Heavier models may not fit in 6 GB VRAM / 16 GB RAM; weights download on first use.")
            if not wan_available().get(mt, False):
                st.caption("Weights for this model are not downloaded yet (or the model list hasn't been refreshed).")
                wan_download_button(mt, names.get(mt, mt), f"vid_dl_{mt}")
            prompt, negative = ui.prompt_box("vid_prompt", "video", target_model=names.get(mt, mt), negative_key="vid_negative")
            c1, c2, c3, c4 = st.columns(4)
            resolution = c1.selectbox("Resolution", preset["resolutions"], key=f"vid_res_{mt}")
            seconds = c2.slider("Duration (s)", 1.0, 10.0, round((preset["frames"] - 1) / preset["fps"], 1), 0.5, key=f"vid_sec_{mt}")
            steps = c3.number_input("Steps (0 = default)", 0, 100, int(preset["steps"] or 0), key=f"vid_steps_{mt}")
            seed = c4.number_input("Seed (-1 = random)", -1, 2**31 - 1, -1, key="vid_seed")
            frames = int(round(seconds * preset["fps"] / 4)) * 4 + 1
            with st.expander("⚙ Advanced"):
                c1, c2 = st.columns(2)
                guidance = c1.number_input("Guidance scale (0 = default)", 0.0, 20.0, float(preset["guidance"] or 0), 0.5, key=f"vid_cfg_{mt}")
                shift = c2.number_input("Flow shift (0 = default)", 0.0, 20.0, float(preset["flow_shift"] or 0), 0.5, key=f"vid_shift_{mt}")
                start_img = st.file_uploader("Start image (only for image-to-video capable models, e.g. Wan 2.2 TI2V)",
                                             type=["png", "jpg", "jpeg", "webp"], key="vid_start")
                st.caption(f"{frames} frames @ {preset['fps']} fps")
            st.info("On an RTX 3050 6 GB, a 5 s 480p clip with Wan 2.1 1.3B takes roughly 5–15 minutes. "
                    "Close other GPU-heavy apps (and browser GPU acceleration) for best results.")
            if st.button("🎬 Generate video locally", type="primary", key="vid_go"):
                if not prompt.strip():
                    st.error("Enter a prompt.")
                else:
                    task = {"model_type": mt, "prompt": prompt, "negative_prompt": negative or None,
                            "resolution": resolution, "video_length": frames, "seed": int(seed)}
                    if steps:
                        task["num_inference_steps"] = int(steps)
                    if guidance:
                        task["guidance_scale"] = float(guidance)
                    if shift:
                        task["flow_shift"] = float(shift)
                    if start_img:
                        task["image_start"] = str(save_upload(start_img))
                        task["image_prompt_type"] = "S"
                    run_wan2gp(task, "video", names.get(mt, mt))
    else:
        category, model = ui.cloud_model_picker(["t2v", "i2v", "v2v"], "cvid")
        prompt, negative = ui.prompt_box("cvid_prompt", "video", target_model=model["name"],
                                         negative_key="cvid_negative" if "negative_prompt" in (model.get("inputs") or {}) else None)
        rendered = ui.media_inputs(catalog.media_fields(model, category), f"cvid_{model['id']}")
        with st.expander("⚙ Model parameters", expanded=True):
            params = ui.render_params(catalog.param_fields(model), f"cvid_{model['id']}")
        if st.button("☁ Generate on MuAPI", type="primary", key="cvid_go"):
            run_cloud(category, model, prompt, negative, params, rendered, "video")

    if st.session_state.get("last_video"):
        st.divider()
        st.subheader("Result")
        ui.show_outputs(st.session_state["last_video"]["files"], st.session_state["last_video"]["urls"], "video", "res_vid")

# ── AUDIO ────────────────────────────────────────────────────────────────────
with tab_aud:
    engine = st.radio("Engine", ["Cloud · MuAPI (music, TTS, SFX)", "Local · Wan2GP audio models"], horizontal=True, key="aud_engine")
    if engine.startswith("Cloud"):
        category, model = ui.cloud_model_picker(["audio"], "caud")
        prompt = ""
        if catalog.has_prompt(model):
            prompt, _ = ui.prompt_box("caud_prompt", "audio", target_model=model["name"], label="Prompt / lyrics")
        rendered = ui.media_inputs(catalog.media_fields(model, category), f"caud_{model['id']}")
        with st.expander("⚙ Model parameters", expanded=True):
            params = ui.render_params(catalog.param_fields(model), f"caud_{model['id']}")
        if st.button("☁ Generate audio on MuAPI", type="primary", key="caud_go"):
            run_cloud(category, model, prompt, "", params, rendered, "audio")
    else:
        if not local_video.installed():
            st.warning("Wan2GP is not installed.")
            wan_install_button("wgp_install_aud")
        else:
            am = wan_models("audio")
            if not am:
                st.info("No Wan2GP audio models listed yet. Refresh the Wan2GP model list in 📦 Models & Engines.")
            else:
                sel = st.selectbox("Wan2GP audio model", am, key="waud_model",
                                   format_func=lambda x: f"{'✅' if local_video.is_available(x) else '⬇'} {x['name']} · {x['model_type']}")
                st.caption("Weights download automatically on first use. Many audio models are large; check VRAM/RAM fit.")
                if not local_video.is_available(sel):
                    wan_download_button(sel["model_type"], sel["name"], f"waud_dl_{sel['model_type']}")
                prompt, _ = ui.prompt_box("waud_prompt", "audio", target_model=sel["name"], label="Text / lyrics / description")
                c1, c2 = st.columns(2)
                dur = c1.number_input("Duration (s, if supported)", 1, 300, 20, key="waud_dur")
                seed = c2.number_input("Seed (-1 = random)", -1, 2**31 - 1, -1, key="waud_seed")
                ref = st.file_uploader("Reference audio (voice cloning / audio guide, optional)",
                                       type=["wav", "mp3", "flac", "m4a"], key="waud_ref")
                if st.button("🎵 Generate audio locally", type="primary", key="waud_go"):
                    task = {"model_type": sel["model_type"], "prompt": prompt, "duration_seconds": int(dur), "seed": int(seed)}
                    if ref:
                        task["audio_guide"] = str(save_upload(ref))
                    run_wan2gp(task, "audio", sel["name"])

    if st.session_state.get("last_audio"):
        st.divider()
        st.subheader("Result")
        ui.show_outputs(st.session_state["last_audio"]["files"], st.session_state["last_audio"]["urls"], "audio", "res_aud")

# ── LIP SYNC & MORE (cloud studios) ──────────────────────────────────────────
with tab_lip:
    st.caption("Cloud studios from the repo: Lip Sync, Body Swap / Recast and Motion Control (MuAPI).")
    category, model = ui.cloud_model_picker(["lipsync", "recast", "motion_control"], "clip")
    prompt = ""
    if catalog.has_prompt(model):
        prompt, _ = ui.prompt_box("clip_prompt", "video", target_model=model["name"], label="Prompt (optional)")
    rendered = ui.media_inputs(catalog.media_fields(model, category), f"clip_{model['id']}")
    with st.expander("⚙ Model parameters", expanded=True):
        params = ui.render_params(catalog.param_fields(model), f"clip_{model['id']}")
    if st.button("☁ Run on MuAPI", type="primary", key="clip_go"):
        run_cloud(category, model, prompt, "", params, rendered, "video")
    if st.session_state.get("last_video") and st.session_state.get("clip_go"):
        ui.show_outputs(st.session_state["last_video"]["files"], st.session_state["last_video"]["urls"], "video", "res_lip")

# ── PROMPT LAB ───────────────────────────────────────────────────────────────
with tab_lab:
    st.subheader("✍️ Prompt Lab")
    st.caption(f"Chat with **{S['llm_provider']}** · `{S['llm_model']}` to brainstorm prompts, then send them to a studio.")
    st.session_state.setdefault("lab_msgs", [])
    templates = {
        "Free chat": "",
        "Image prompt": "Write 3 detailed text-to-image prompt variations for: ",
        "Video shot": "Write a 5-second cinematic video prompt with camera motion for: ",
        "Storyboard (5 shots)": "Write a 5-shot storyboard; for each shot give a video prompt (one line each) for: ",
        "Song / music": "Write a music generation prompt (genre, mood, BPM, instruments) and short lyrics for: ",
    }
    c1, c2 = st.columns([1, 3])
    tpl = c1.selectbox("Template", list(templates), key="lab_tpl")
    for msg in st.session_state["lab_msgs"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_msg = st.chat_input("Describe your idea...")
    if user_msg:
        content = templates[tpl] + user_msg
        st.session_state["lab_msgs"].append({"role": "user", "content": content})
        with st.chat_message("user"):
            st.markdown(content)
        kw = ui.llm_call_kwargs()
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    reply = llm.chat(kw["provider"], kw["model"],
                                     [{"role": "system", "content": llm.SYSTEM_PROMPTS["chat"]}] + st.session_state["lab_msgs"],
                                     api_key=kw["api_key"], base_url=kw["base_url"])
                st.markdown(reply)
                st.session_state["lab_msgs"].append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.error(f"LLM error: {exc}")
    if st.session_state["lab_msgs"]:
        last = next((m["content"] for m in reversed(st.session_state["lab_msgs"]) if m["role"] == "assistant"), "")
        edited = st.text_area("Prompt to send", value=last, key="lab_send_text", height=120)

        def send_to(key, text):
            st.session_state[key] = text

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.button("→ Local image", on_click=send_to, args=("img_prompt", edited))
        c2.button("→ Cloud image", on_click=send_to, args=("cimg_prompt", edited))
        c3.button("→ Local video", on_click=send_to, args=("vid_prompt", edited))
        c4.button("→ Cloud video", on_click=send_to, args=("cvid_prompt", edited))
        c5.button("🗑 Clear chat", on_click=lambda: st.session_state.update(lab_msgs=[]))

# ── MODELS & ENGINES ─────────────────────────────────────────────────────────
with tab_models:
    st.subheader("📦 Models & Engines")
    st.caption(f"Local model folder (shared with the desktop app via `OPEN_GENERATIVE_AI_LOCAL_AI_DIR`): `{config.LOCAL_AI_DIR}`")

    st.markdown("### 1 · sd.cpp image engine")
    if local_image.binary_installed():
        st.success(f"Installed: `{config.SD_CLI}`")
    else:
        sd_install_button("sd_install")

    st.markdown("### 2 · Local image models (repo catalog)")
    for m in local_image.list_models():
        c1, c2, c3 = st.columns([4, 1, 1])
        tag = " ⭐ default" if m["id"] == config.DEFAULT_LOCAL_IMAGE_MODEL else ""
        c1.markdown(f"**{m['name']}**{tag} · {m['sizeGB']} GB{' (+2.7 GB text encoder/VAE)' if m.get('requiresAuxiliary') else ''}  \n"
                    f"<small>{m['description']}</small>", unsafe_allow_html=True)
        if m["ready"]:
            c2.success("Ready")
            if c3.button("Delete", key=f"del_{m['id']}"):
                local_image.delete_model(m["id"])
                st.rerun()
        else:
            c2.info("Partial" if m["partial"] else "Not downloaded")
            if c3.button("⬇ Download", key=f"dl_{m['id']}"):
                bar = st.progress(0.0)
                try:
                    local_image.download_model(m["id"], lambda t, p: bar.progress(min(p, 1.0), text=t))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    st.markdown("### 3 · Wan2GP (local video / image / audio)")
    if not local_video.installed():
        st.warning(f"Not installed at `{config.WAN2GP_DIR}`.")
        wan_install_button("wgp_install")
    else:
        c1, c2, c3 = st.columns(3)
        if c1.button("Check PyTorch + CUDA", key="wgp_torch"):
            good, info = local_video.torch_ok()
            (st.success if good else st.error)(info)
        prof = c2.selectbox("Memory profile", [4, 5, 2, 1, 3], index=[4, 5, 2, 1, 3].index(int(S["wan2gp_profile"])),
                            help="4 = default/recommended, 5 = minimum RAM", key="wgp_profile")
        att = c3.selectbox("Attention", ["sdpa", "sage2", "sage", "flash"],
                           index=["sdpa", "sage2", "sage", "flash"].index(S["wan2gp_attention"]), key="wgp_att",
                           help="sdpa works everywhere; sage2 is faster but needs SageAttention installed")
        if (prof, att) != (S["wan2gp_profile"], S["wan2gp_attention"]):
            S.update(wan2gp_profile=prof, wan2gp_attention=att)
            persist()
        c1, c2 = st.columns(2)
        if c1.button("🔄 Refresh Wan2GP model list", key="wgp_refresh"):
            with st.spinner("Loading WanGP runtime and reading its model catalog (1-5 min, slowest the first time)..."):
                try:
                    local_video.list_models(S, refresh=True)
                    st.success("Model list refreshed.")
                except Exception as exc:
                    st.error(str(exc)[-2000:])
        if local_video.webui_running():
            c2.markdown(f"🟢 Full Wan2GP UI running: [http://127.0.0.1:{local_video.WEBUI_PORT}](http://127.0.0.1:{local_video.WEBUI_PORT})")
        elif c2.button("Launch full Wan2GP web UI (advanced)", key="wgp_webui"):
            local_video.start_webui(S)
            st.info(f"Starting... open http://127.0.0.1:{local_video.WEBUI_PORT} in ~1 minute.")
        try:
            all_wm = local_video.list_models(S)
        except Exception:
            all_wm = []
        if all_wm:
            q = st.text_input("Filter Wan2GP models", key="wgp_filter")
            rows = [{"available": "✅" if local_video.is_available(x) else "⬇",
                     "name": x["name"], "model_type": x["model_type"], "family": x.get("family"),
                     "output": ", ".join(x["main_output"]) if isinstance(x.get("main_output"), list) else x.get("main_output"),
                     "inputs": ", ".join(x.get("inputs") or [])}
                    for x in all_wm if not q or q.lower() in (x["name"] + x["model_type"]).lower()]
            st.dataframe(rows, width="stretch", hide_index=True, height=300)
            st.caption("⬇ models are downloaded automatically the first time you generate with them, "
                       "or download one now:")
            missing = sorted(all_wm, key=lambda x: local_video.is_available(x))
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("Wan2GP model to download", missing, key="wgp_dl_pick", label_visibility="collapsed",
                                format_func=lambda x: f"{'✅' if local_video.is_available(x) else '⬇'} {x['name']} · {x['model_type']}")
            with c2:
                wan_download_button(pick["model_type"], pick["name"], "wgp_dl_go")
        else:
            st.caption("Recommended video models (click 'Refresh Wan2GP model list' to see all):")
            for mt, preset in local_video.RECOMMENDED.items():
                wan_download_button(mt, preset["label"], f"wgp_dl_rec_{mt}")

    st.markdown("### 4 · Ollama (local LLM for prompts)")
    fv = st.checkbox("Unload Ollama models from VRAM before local image/video generation (recommended on 6 GB)",
                     value=S.get("free_vram_before_local", True), key="free_vram_opt")
    if fv != S.get("free_vram_before_local", True):
        S["free_vram_before_local"] = fv
        persist()
    if llm.ollama_running():
        st.success("Ollama is running. Installed models: " + (", ".join(llm.ollama_models()) or "none"))
        c1, c2 = st.columns([3, 1])
        pull = c1.text_input("Pull a model", value="qwen3:4b", key="ollama_pull",
                             help="Good small choices for 6 GB VRAM: qwen3:4b, llama3.2:3b, gemma3:4b")
        if c2.button("Pull", key="ollama_pull_go"):
            bar = st.progress(0.0)
            try:
                for ev in llm.ollama_pull(pull):
                    if ev.get("total"):
                        bar.progress(min(ev.get("completed", 0) / ev["total"], 1.0), text=ev.get("status", ""))
                st.success(f"Pulled {pull}")
            except Exception as exc:
                st.error(str(exc))
    else:
        st.warning("Ollama is not running. Install it from https://ollama.com/download (or `winget install Ollama.Ollama`) and start it.")

    st.markdown("### 5 · Cloud model catalog (from the repo)")
    cat = catalog.load_catalog()
    st.write({catalog.CATEGORY_LABELS[k]: len(v) for k, v in cat["muapi"].items()})
    st.caption(f"Exported {cat['generated_at']} from `{cat['repo_dir']}`")
    if st.button("🔄 Re-export catalog from repo", key="cat_refresh"):
        try:
            st.code(catalog.refresh_catalog())
        except Exception as exc:
            st.error(str(exc))

# ── HISTORY ──────────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("🕘 History")
    kinds = st.multiselect("Show", ["image", "video", "audio"], default=["image", "video", "audio"], key="hist_kinds")
    items = [h for h in history.load(100) if h["kind"] in kinds]
    if not items:
        st.info(f"Nothing yet. Outputs are saved in `{config.OUTPUTS_DIR}`.")
    for i, h in enumerate(items[:30]):
        with st.container(border=True):
            st.markdown(f"**{h['ts']}** · {h['kind']} · `{h['engine']}` · **{h['model']}**")
            if h.get("prompt"):
                st.caption(h["prompt"][:300])
            files = [f for f in h["files"] if Path(f).exists()]
            ui.show_outputs(files, h.get("urls") if not files else None, h["kind"], f"hist{i}")

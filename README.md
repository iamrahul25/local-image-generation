# Open Generative AI – Streamlit Studio

A Streamlit web UI on top of [Anil-matcha/Open-Generative-AI](https://github.com/Anil-matcha/Open-Generative-AI).
It generates **images, videos and audio**, either **locally on your GPU** or through the repo's **cloud catalog of 537 models** (MuAPI), and uses a **local or API LLM** to write and improve your prompts.

It is built for this machine: Windows 11, Ryzen 5 8645HS, 16 GB RAM, **RTX 3050 Laptop 6 GB**.

| What | Engine | Default model | Where it runs |
| --- | --- | --- | --- |
| Local images | [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (the repo's bundled engine) | **Z-Image Turbo** (Q4 GGUF, 8 steps) | Your GPU |
| Local video | [Wan2GP](https://github.com/deepbeepmeep/Wan2GP) (the repo's video engine) | **Wan 2.1 T2V 1.3B** (480p) | Your GPU |
| Local audio / extra local image models | Wan2GP | Any Wan2GP model (downloads on first use) | Your GPU |
| Prompt LLM | Ollama (local) or OpenAI / Gemini / Groq / OpenRouter / DeepSeek / Claude / any OpenAI-compatible API | `qwen3:4b-instruct` on Ollama | Local or cloud |
| Cloud generation | MuAPI (the repo's API layer) | 537 models: Flux, Midjourney, Nano Banana, Kling, Veo, Sora, Seedance, Suno, lip sync... | Cloud (paid, needs key) |

---

## 1. How it connects to the repo

```
img-gen/
├── app.py                      ← Streamlit UI (run this)
├── studio/                     ← Python modules used by the UI
│   ├── catalog.py              ← cloud catalog + schema → form/payload logic
│   ├── muapi.py                ← Python port of the repo's packages/studio/src/muapi.js
│   ├── local_image.py          ← sd.cpp runner (mirrors electron/lib/localInference.js)
│   ├── local_video.py          ← Wan2GP runner (the repo's second local engine)
│   ├── llm.py                  ← Ollama / OpenAI-compatible / Anthropic prompt helpers
│   ├── ui.py, history.py, sysinfo.py, config.py, downloads.py
├── scripts/
│   ├── export_catalog.mjs      ← reads the repo's models.js + modelCatalog.js → data/catalog.json
│   ├── wan2gp_worker.py        ← runs inside Wan2GP's venv, drives WanGP's Python API
│   └── setup_local.py          ← installs sd.cpp + downloads local image models
├── Open-Generative-AI/         ← the cloned repo (source of all model catalogs)
├── engines/Wan2GP/             ← Wan2GP + its own Python 3.11 / PyTorch CUDA venv
├── local-ai/                   ← sd.cpp binaries (bin/) and weights (models/)
├── outputs/                    ← every generated image / video / audio + history.jsonl
├── setup.bat                   ← one-time install
└── run.bat                     ← start the app
```

* **Cloud models** come straight from the repo: `scripts/export_catalog.mjs` imports
  `Open-Generative-AI/packages/studio/src/models.js`. The UI builds each model's form (aspect ratio,
  duration, resolution, seed, uploads...) from that schema, then calls MuAPI the same way the repo does
  (submit, then poll `/api/v1/predictions/{id}/result`).
* **Local image models** use the repo's own sd.cpp catalog (`electron/lib/modelCatalog.js`): Z-Image
  Turbo/Base, DreamShaper 8, Realistic Vision 5.1, Anything v5 and SDXL. The CLI flags and folder layout
  are the same as the desktop app's.
* **Local video** uses Wan2GP, the engine the repo integrates for video. Instead of the desktop app's
  Gradio bridge, this app calls WanGP's official in-process Python API, which is more reliable and gives
  real progress updates.
* Two separate Python environments are used on purpose: the UI runs on Python 3.12 with no PyTorch, and
  Wan2GP runs on Python 3.11 with PyTorch CUDA. Each video job runs in its own process, so all RAM and VRAM
  are freed when it finishes. This matters on a 16 GB machine.

---

## 2. Install (one time)

Requirements: **Python 3.10+**, **Git**, **Node.js 18+**, an NVIDIA driver (you have 591.80 / CUDA 13.1),
and about **30 GB** of free disk.

```bat
cd C:\all-projects\img-gen
setup.bat
```

`setup.bat` does the following:

1. Creates `.venv` (Python 3.12) and installs Streamlit and the other small dependencies.
2. Clones Open-Generative-AI (with submodules, if missing) and exports its model catalog.
3. Installs the **sd.cpp CUDA 12** engine and downloads **Z-Image Turbo** (5.2 GB, including the
   Qwen3-4B text encoder and the FLUX VAE).
4. Clones **Wan2GP**, creates its Python 3.11 venv, and installs PyTorch 2.10 (CUDA 13),
   Wan2GP's requirements and Triton.
5. Pulls `qwen3:4b-instruct` if Ollama is installed.

The **Wan 2.1 1.3B** video weights (about 8 GB, including the text encoder) are downloaded by Wan2GP
automatically the **first time** you generate a local video. That first run is slow. To avoid the wait,
click **⬇ Download** next to the model in the Video tab or in **📦 Models & Engines** beforehand.

Model weights, engines and outputs are **not** in git (see `.gitignore`). After cloning the repo you can
either run `setup.bat`, or just run `run.bat` and use the **⬇ Install / ⬇ Download** buttons in the app
(sd.cpp engine, image models, Wan2GP, and Wan2GP video/image/audio models).

Ollama (local LLM): `winget install Ollama.Ollama`, then `ollama pull qwen3:4b-instruct`. You can also pull
models from inside the app.

### Manual steps (if you prefer)

```bat
python -m pip install --user uv
python -m uv venv .venv --python 3.12
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
node scripts\export_catalog.mjs
.venv\Scripts\python scripts\setup_local.py --all                       & rem sd.cpp + Z-Image Turbo
.venv\Scripts\python scripts\setup_local.py --image-model dreamshaper-8  & rem add another model later
```

---

## 3. Run

```bat
run.bat
```

Then open **http://localhost:8501**. The equivalent command is `.venv\Scripts\python -m streamlit run app.py`.

---

## 4. Using the app

### Sidebar
* **System**: CPU, RAM, GPU name, VRAM use, temperature and free disk space.
* **Engines**: green/red status for sd.cpp, the local image model, Wan2GP, Ollama and the MuAPI key.
* **MuAPI key**: paste it here to enable the cloud models, and use *Check balance* to see your credits.
  The key is stored in `data/settings.json`, which is git-ignored.
* **Prompt LLM**: pick the provider (Ollama, LM Studio, OpenAI, Gemini, Groq, OpenRouter, DeepSeek, Claude,
  or a custom OpenAI-compatible endpoint), the model, an optional base URL and an API key.

### 🖼 Image tab
Pick an engine:
* **Local · sd.cpp**: choose any model from the repo's local catalog (Z-Image Turbo is the default).
  * Text→Image or Image→Image (upload an init image and set the strength).
  * Aspect ratio, base size, steps, CFG, sampler, scheduler, seed and the number of images.
  * **Low-VRAM mode** (on by default) adds `--offload-to-cpu --vae-tiling`, which keeps 1024 px
    generation inside 6 GB. You can also add extra `sd-cli` flags.
* **Local · Wan2GP image models**: Z-Image, Flux, Qwen Image, Krea and others via Wan2GP. Weights
  download on first use.
* **Cloud · MuAPI**: text-to-image (78 models) and image-to-image/edit (76 models). Filter by provider,
  search, upload reference images, and set every model-specific parameter.

Every prompt box has **✨ Enhance with LLM**, **🚫 Suggest negative** and **↩ Undo** buttons.

### 🎬 Video tab
* **Local · Wan2GP**: Wan 2.1 T2V 1.3B is the default. The list also includes Wan 2.2 TI2V 5B FastWan,
  LTX Video, and every other video model your Wan2GP installation knows about.
  * You can set the resolution, duration (converted to valid frame counts), steps, guidance, flow shift,
    seed, and an optional start image for image-to-video models.
  * Expect about **5–15 minutes per 5 s 480p clip** on the RTX 3050. A live progress bar and log are shown.
* **Cloud · MuAPI**: text-to-video (105), image-to-video (173) and video-to-video/effects (67) models,
  including Kling, Veo, Sora, Seedance, Hailuo, Runway and Wan.

### 🎵 Audio tab
* **Cloud · MuAPI**: Suno music, remix/extend, MiniMax speech, sound effects and more.
* **Local · Wan2GP audio models**: Qwen3 TTS, IndexTTS2, ACE-Step, Stable Audio and others, with optional
  reference audio for voice cloning. Weights download on first use; check that the model fits in memory.

### 🗣 Lip Sync & More
The repo's cloud studios: **Lip Sync** (portrait + audio, or video + audio), **Body Swap / Recast** and
**Motion Control**.

### ✍️ Prompt Lab
Chat with your LLM using templates (image prompt, video shot, 5-shot storyboard, song). Then send the
result to the local or cloud image/video prompt with one click.

### 📦 Models & Engines
* Install or reinstall sd.cpp (CUDA, Vulkan or CPU build).
* Download or delete any local image model from the repo catalog.
* Install Wan2GP with one click, and download any Wan2GP model's weights ahead of time.
* Wan2GP: check PyTorch/CUDA, choose the **memory profile** (4 is the default; 5 uses the least RAM) and
  the **attention** backend, refresh the Wan2GP model list (takes 1–5 minutes; the result is cached), and optionally launch the **full Wan2GP web UI**
  on port 7860.
* Ollama: see installed models and pull new ones (`qwen3:4b-instruct`, `llama3.2:3b`, `gemma3:4b`...).
* Re-export the cloud catalog after you `git pull` the repo.

### 🕘 History
Browse all past generations. Files are in `outputs/images`, `outputs/videos` and `outputs/audios`, and
`outputs/history.jsonl` records the prompt and settings for each one.

---

## 5. Adding more models later

| Want | How |
| --- | --- |
| Another sd.cpp image model | **📦 Models & Engines → Download**, or `setup_local.py --image-model <id>` (`dreamshaper-8`, `realistic-vision-v51`, `anything-v5`, `stable-diffusion-xl-base`, `z-image-base`) |
| A new sd.cpp model not in the repo catalog | Add an entry to `Open-Generative-AI/electron/lib/modelCatalog.js` (id, filename, downloadUrl, type, defaults), then click **Re-export catalog** |
| Another local video/audio/image model | Select it in the Video, Audio or Wan2GP-image dropdown and click **⬇ Download** (or let Wan2GP download it on first use). Click **Refresh Wan2GP model list** after updating Wan2GP |
| Newer cloud models | `cd Open-Generative-AI && git pull`, then **Re-export catalog** |
| Another LLM | `ollama pull <model>`, or pick any API provider in the sidebar |

Share the local model folder with the repo's desktop app by setting
`OPEN_GENERATIVE_AI_LOCAL_AI_DIR=C:\all-projects\img-gen\local-ai` (in System → Environment Variables).
Both apps then see the same sd.cpp binaries and weights.

---

## 6. Tips for a 6 GB GPU / 16 GB RAM

* Keep **Low-VRAM mode** on for Z-Image and SDXL. If an image fails with an out-of-memory error, lower the
  base size to 768.
* For video, close games and GPU-heavy apps, and consider turning off hardware acceleration in your browser.
  This frees 1–2 GB of VRAM.
* Keep Wan2GP on memory profile **4**. Switch to **5** if Windows runs out of RAM.
* Don't run a local image job and a local video job at the same time.
* Ollama keeps its model in VRAM for about 5 minutes after use. The app unloads it automatically before every
  local image/video job (toggle: **📦 Models & Engines → Ollama → Free VRAM before local generation**).
* Use `qwen3:4b-instruct`, not `qwen3:4b`. The plain tag is a "thinking" model and takes 10x longer to enhance a prompt.
* Speed-ups for later: install SageAttention 2 in the Wan2GP venv and choose `sage2` attention.
  See `engines/Wan2GP/docs/INSTALLATION.md`.

## 7. Troubleshooting

| Problem | Fix |
| --- | --- |
| `sd-cli exited with code ...` | Read the log tail shown in the UI. Out-of-memory: lower the size or keep Low-VRAM on. Missing DLLs: reinstall sd.cpp in Models & Engines |
| Video stuck on "Loading WanGP runtime" | The first run downloads about 8 GB of weights. Watch the log in the status box, or `engines/Wan2GP/ckpts` |
| Wan2GP import errors | Re-run step 5 of `setup.bat`, or `engines\Wan2GP\.venv\Scripts\python -m pip install -r engines\Wan2GP\requirements.txt` |
| Enhance fails with Ollama | Start the Ollama app, then `ollama pull qwen3:4b-instruct` |
| Cloud error 401/403 | Check your MuAPI key and balance |
| Port 8501 in use | `.venv\Scripts\python -m streamlit run app.py --server.port 8502` |
| `cannot make enough memory available` (sd.cpp) | Another program holds VRAM (often Ollama). Keep the free-VRAM toggle on, or run `ollama stop <model>` |
| Model download stops / `Connection broken: IncompleteRead` | Just retry. Image downloads resume from `.part` files and video jobs retry automatically (Hugging Face resumes partial files). The app sets `HF_HUB_DISABLE_XET=1` because the Xet downloader stalled at 0 bytes on this machine |

Licenses: Open-Generative-AI (MIT), stable-diffusion.cpp (MIT), Wan2GP (see its license and terms; this
app uses **WanGP** as its local video engine), and each model has its own license.

"""Quick end-to-end check of the local video engine (also pre-downloads the default model).

    .venv\\Scripts\\python scripts\\smoke_test_video.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config, local_video  # noqa: E402


def on_event(ev: dict) -> None:
    if ev["event"] == "progress":
        print(f"[progress] {ev.get('phase')} {ev.get('progress')}% step {ev.get('step')}/{ev.get('total')}", flush=True)
    elif ev["event"] in ("status", "log"):
        print(f"[{ev['event']}] {ev.get('text', '')}", flush=True)


if __name__ == "__main__":
    started = time.time()
    task = {
        "model_type": config.DEFAULT_LOCAL_VIDEO_MODEL,
        "prompt": "A red fox walking through fresh snow in a pine forest, soft morning light, cinematic",
        "resolution": "832x480", "video_length": 17, "num_inference_steps": 10, "seed": 42,
    }
    done = local_video.generate(config.load_settings(), task, on_event)
    print(f"DONE in {time.time() - started:.0f}s -> {done['files']}")

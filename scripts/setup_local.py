"""One-shot installer for the local engines and default models.

Usage (from the project root, with the app venv):
    .venv\\Scripts\\python scripts\\setup_local.py --all
    .venv\\Scripts\\python scripts\\setup_local.py --sdcpp
    .venv\\Scripts\\python scripts\\setup_local.py --image-model dreamshaper-8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from studio import config, local_image  # noqa: E402

_last = {"label": None, "pct": -1}


def progress(label: str, frac: float) -> None:
    pct = int(frac * 100)
    if label != _last["label"]:
        print()
        _last["label"] = label
        _last["pct"] = -1
    if pct != _last["pct"]:
        _last["pct"] = pct
        print(f"\r{label}: {pct:3d}%", end="", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="sd.cpp engine + default local image model")
    ap.add_argument("--sdcpp", action="store_true", help="install the sd.cpp CUDA engine")
    ap.add_argument("--backend", default="cuda12", help="sd.cpp build: cuda12 | vulkan | cpu")
    ap.add_argument("--image-model", default=None, help="sd.cpp catalog id to download")
    args = ap.parse_args()

    if args.all or args.sdcpp:
        print(f"Installing sd.cpp ({args.backend}) into {config.BIN_DIR}")
        local_image.install_binary(progress, backend=args.backend)
        print(f"\nsd.cpp ready: {config.SD_CLI}")

    model_id = args.image_model or (config.DEFAULT_LOCAL_IMAGE_MODEL if args.all else None)
    if model_id:
        print(f"Downloading local image model '{model_id}' into {config.MODELS_DIR}")
        local_image.download_model(model_id, progress)
        print("\nModel ready.")


if __name__ == "__main__":
    main()

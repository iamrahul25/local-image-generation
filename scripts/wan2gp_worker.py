"""Bridge between the Streamlit app and WanGP's in-process Python API.

Runs inside the Wan2GP virtualenv (Python 3.11 + PyTorch). Every message for the
Streamlit app is one line on stdout prefixed with MARKER followed by JSON.

    python wan2gp_worker.py --root <Wan2GP dir> list-models [--out file.json]
    python wan2gp_worker.py --root <Wan2GP dir> defaults <model_type>
    python wan2gp_worker.py --root <Wan2GP dir> download <model_type>
    python wan2gp_worker.py --root <Wan2GP dir> generate --settings job.json --output-dir <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

MARKER = "@@WGP@@ "
# WanGP captures sys.stdout, so keep a private handle to the real stdout.
_real_stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)


def emit(event: str, **data) -> None:
    _real_stdout.write(MARKER + json.dumps({"event": event, **data}, default=str) + "\n")
    _real_stdout.flush()


def open_session(root: Path, profile: str, attention: str, output_dir: str | None = None, verbose: bool = False):
    os.chdir(root)
    sys.path.insert(0, str(root))
    from shared.api import init  # noqa: WPS433 (import inside WanGP root)

    cli_args = ["--attention", attention, "--profile", str(profile)]
    return init(root=root, output_dir=output_dir, cli_args=cli_args, console_output=verbose)


def cmd_list_models(session, out: str | None) -> None:
    records = session.list_model_metadata(include_availability=True)
    slim = []
    for r in records:
        meta = r.get("metadata", r)
        slim.append({
            "model_type": r.get("model_type") or meta.get("model_type"),
            "name": r.get("name") or meta.get("name"),
            "family": meta.get("family_label") or meta.get("family"),
            "main_output": meta.get("main_output"),
            "inputs": meta.get("inputs"),
            "capabilities": meta.get("capabilities"),
            "availability": r.get("availability") or meta.get("availability"),
        })
    if out:
        Path(out).write_text(json.dumps(slim, indent=1, default=str), encoding="utf-8")
    emit("models", count=len(slim), models=slim if not out else None, file=out)


def cmd_defaults(session, model_type: str) -> None:
    emit("defaults", model_type=model_type, settings=session.get_default_settings(model_type),
         schema=session.get_model_schema(model_type))


class _DownloadsDone(Exception):
    pass


def cmd_download(session, model_type: str) -> None:
    """Fetch every weight file a model needs without loading it onto the GPU."""
    if session.get_model_def(model_type) is None:
        raise ValueError(f"Unknown model_type: {model_type}")
    wgp = session._ensure_runtime().module
    original_check = wgp.check_download_cancelled

    # load_models() calls this right after its last download and before touching torch/VRAM.
    def stop_after_downloads(gen=None):
        original_check(gen)
        raise _DownloadsDone()

    emit("status", text=f"Downloading weights for {model_type}")
    started = time.time()
    wgp.check_download_cancelled = stop_after_downloads
    try:
        wgp.load_models(model_type)
    except _DownloadsDone:
        pass
    finally:
        wgp.check_download_cancelled = original_check
    emit("downloaded", model_type=model_type, availability=session.get_model_availability(model_type),
         elapsed=round(time.time() - started, 1))


def cmd_generate(session, settings_file: str) -> None:
    settings = json.loads(Path(settings_file).read_text(encoding="utf-8"))
    model_type = settings["model_type"]
    merged = session.get_default_settings(model_type)
    merged.update({k: v for k, v in settings.items() if v is not None})
    emit("status", text=f"Starting {model_type}")

    started = time.time()
    job = session.submit_task(merged)
    last_emit = 0.0
    for ev in job.events.iter(timeout=0.5):
        kind = ev.kind
        data = ev.data
        if kind == "progress" and data is not None:
            now = time.time()
            if now - last_emit > 0.5:
                last_emit = now
                emit("progress", phase=getattr(data, "phase", ""), status=getattr(data, "status", ""),
                     progress=getattr(data, "progress", 0), step=getattr(data, "current_step", None),
                     total=getattr(data, "total_steps", None))
        elif kind == "status":
            emit("status", text=str(data))
        elif kind == "stream" and data is not None:
            text = getattr(data, "text", str(data)).rstrip()
            if text:
                emit("log", text=text[-500:])
        if job.done:
            break

    result = job.result()
    emit("done", success=result.success, files=list(result.generated_files),
         errors=[str(e) for e in result.errors], elapsed=round(time.time() - started, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--profile", default="4")
    ap.add_argument("--attention", default="sdpa")
    ap.add_argument("--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list-models")
    p_list.add_argument("--out")
    p_def = sub.add_parser("defaults")
    p_def.add_argument("model_type")
    p_dl = sub.add_parser("download")
    p_dl.add_argument("model_type")
    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--settings", required=True)
    p_gen.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    try:
        emit("status", text="Loading WanGP runtime...")
        session = open_session(Path(args.root), args.profile, args.attention,
                               getattr(args, "output_dir", None), args.verbose)
        if args.cmd == "list-models":
            cmd_list_models(session, args.out)
        elif args.cmd == "defaults":
            cmd_defaults(session, args.model_type)
        elif args.cmd == "download":
            cmd_download(session, args.model_type)
        else:
            cmd_generate(session, args.settings)
    except Exception as exc:  # report everything back to the UI
        import traceback
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-4000:])
        sys.exit(1)
    finally:
        _real_stdout.flush()
    os._exit(0)  # WanGP leaves non-daemon threads alive


if __name__ == "__main__":
    main()

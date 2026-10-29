import platform
import shutil
import subprocess
from functools import lru_cache

import psutil

from . import config


@lru_cache(maxsize=1)
def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return out.stdout.strip() or platform.processor()
    except (OSError, subprocess.SubprocessError):
        return platform.processor()


def gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        name, total, used, driver, util, temp = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "vram_total_mb": int(total), "vram_used_mb": int(used), "driver": driver,
                "util": int(util), "temp": int(temp)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def port_listening(port: int) -> bool:
    """Check for a local listener without connecting (refused connects take ~2s on Windows)."""
    try:
        return any(c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN
                   for c in psutil.net_connections(kind="tcp"))
    except (psutil.AccessDenied, OSError):
        return True  # can't tell; let the caller try a real request


def snapshot() -> dict:
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(config.ROOT)
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu": cpu_name(), "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_gb": round(vm.total / 2**30, 1), "ram_used_gb": round(vm.used / 2**30, 1),
        "disk_free_gb": round(disk.free / 2**30, 1), "gpu": gpu(),
    }

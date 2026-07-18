# tools/collect_environment.py
# ------------------------------------------------------------
# Collect experiment environment info for Section 4.2.1
# Outputs:
#   - data/interim/experiment_environment.json
#   - data/interim/experiment_environment.txt
# ------------------------------------------------------------
from __future__ import annotations
import os
import sys
import json
import platform
import subprocess
from datetime import datetime

def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return None

def _pkg_version(name: str) -> str | None:
    # Try import-based version first
    try:
        mod = __import__(name)
        v = getattr(mod, "__version__", None)
        if v is not None:
            return str(v)
    except Exception:
        pass

    # Fallback: importlib.metadata (Python 3.8+)
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None

def _torch_info() -> dict:
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            info["cuda_device_count"] = int(n)
            devices = []
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "index": int(i),
                    "name": props.name,
                    "total_memory_GB": round(props.total_memory / (1024**3), 3),
                    "multi_processor_count": int(getattr(props, "multi_processor_count", -1)),
                })
            info["cuda_devices"] = devices
        else:
            info["cuda_device_count"] = 0
            info["cuda_devices"] = []
        # cuDNN
        try:
            info["cudnn_enabled"] = bool(torch.backends.cudnn.enabled)
            info["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
        except Exception:
            info["cudnn_enabled"] = None
            info["cudnn_version"] = None
    except Exception as e:
        info["torch_error"] = str(e)
    return info

def _cpu_info() -> dict:
    info = {}
    info["machine"] = platform.machine()
    info["processor"] = platform.processor() or None
    # logical/physical cores (best effort)
    try:
        import psutil
        info["cpu_logical_cores"] = int(psutil.cpu_count(logical=True) or 0)
        info["cpu_physical_cores"] = int(psutil.cpu_count(logical=False) or 0)
        vm = psutil.virtual_memory()
        info["ram_total_GB"] = round(vm.total / (1024**3), 3)
        info["ram_available_GB"] = round(vm.available / (1024**3), 3)
    except Exception:
        info["cpu_logical_cores"] = None
        info["cpu_physical_cores"] = None
        info["ram_total_GB"] = None
        info["ram_available_GB"] = None
    return info

def _lib_versions() -> dict:
    libs = [
        # Core
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        # Graph / DL
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("torch_sparse", "torch_sparse"),
        ("torch_scatter", "torch_scatter"),
        ("torch_cluster", "torch_cluster"),
        # Geo
        ("geopandas", "geopandas"),
        ("shapely", "shapely"),
        ("pyproj", "pyproj"),
        ("rtree", "rtree"),
        # Plot
        ("matplotlib", "matplotlib"),
    ]
    out = {}
    for pip_name, import_name in libs:
        out[pip_name] = _pkg_version(import_name)
    return out

def main():
    os.makedirs("data/interim", exist_ok=True)

    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
        },
        "cpu": _cpu_info(),
        "torch": _torch_info(),
        "libs": _lib_versions(),
        "drivers": {
            "nvidia_smi": _safe_run(["nvidia-smi"])  # may be None on CPU machines
        }
    }

    # Write JSON
    out_json = "data/interim/experiment_environment.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(env, f, ensure_ascii=False, indent=2)

    # Write TXT (paper-ready)
    out_txt = "data/interim/experiment_environment.txt"
    lines = []
    lines.append(f"Timestamp: {env['timestamp']}")
    lines.append(f"Python: {env['python_version']}")
    lines.append(f"Executable: {env['python_executable']}")
    p = env["platform"]
    lines.append(f"OS: {p['system']} {p['release']} ({p['platform']})")
    c = env["cpu"]
    lines.append(f"CPU: {c.get('processor') or 'N/A'} | logical cores={c.get('cpu_logical_cores')} | RAM={c.get('ram_total_GB')} GB")
    t = env["torch"]
    lines.append(f"PyTorch: {t.get('torch_version')} | CUDA available={t.get('cuda_available')} | torch CUDA={t.get('torch_cuda_version')} | cuDNN={t.get('cudnn_version')}")
    if t.get("cuda_devices"):
        for d in t["cuda_devices"]:
            lines.append(f"GPU[{d['index']}]: {d['name']} | VRAM={d['total_memory_GB']} GB | SMs={d.get('multi_processor_count')}")
    lines.append("")
    lines.append("Key libraries:")
    for k, v in env["libs"].items():
        if v is not None:
            lines.append(f"- {k}: {v}")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Saved:", out_json)
    print("✅ Saved:", out_txt)
    print("\n".join(lines[:12]))  # short preview

if __name__ == "__main__":
    main()
# Environment & Reproducibility Report
# Writes env_report.json, env_report.txt, pip_freeze.txt to:
# /ceph/lfertig/Thesis/notebook/llm-peft-ppm/results

import os, sys, platform, subprocess, json
from datetime import datetime

OUT_DIR = "/ceph/lfertig/Thesis/notebook/llm-peft-ppm/results"
os.makedirs(OUT_DIR, exist_ok=True)

def run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"<unavailable: {e}>"

def get_pkg_version(dist_name: str) -> str:
    try:
        from importlib.metadata import version
        return version(dist_name)
    except Exception:
        try:
            mod = __import__(dist_name)
            return getattr(mod, "__version__", "<no __version__>")
        except Exception as e:
            return f"<not available: {e}>"

report = {
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "python": sys.version.replace("\n", " "),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    },
    "os_release": run("cat /etc/os-release"),
    "kernel": run("uname -a"),
    "hardware": {
        "cpu_lscpu": run("lscpu"),
        "ram_free_h": run("free -h"),
        "gpu_nvidia_smi": run("nvidia-smi"),
        "gpu_nvidia_smi_query": run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"),
    },
    "packages": {
        "numpy": get_pkg_version("numpy"),
        "pandas": get_pkg_version("pandas"),
        "scikit-learn": get_pkg_version("scikit-learn"),
        "matplotlib": get_pkg_version("matplotlib"),
        "seaborn": get_pkg_version("seaborn"),
        "torch": get_pkg_version("torch"),
        "transformers": get_pkg_version("transformers"),
        "peft": get_pkg_version("peft"),
        "torchmetrics": get_pkg_version("torchmetrics"),
        "datasets": get_pkg_version("datasets"),
        # baselines / repos (best-effort; names may differ in your env)
        "skpm": get_pkg_version("skpm"),
        "tabpfn": get_pkg_version("tabpfn"),
        "tabpfn-extensions": get_pkg_version("tabpfn-extensions"),
        "sap_rpt_oss": get_pkg_version("sap_rpt_oss"),
        "wandb": get_pkg_version("wandb"),
        "pydantic": get_pkg_version("pydantic"),
    },
    "wandb_env": {k: os.environ.get(k) for k in [
        "WANDB_PROJECT", "WANDB_ENTITY", "WANDB_MODE", "WANDB_RUN_ID", "WANDB_DIR", "WANDB_CACHE_DIR"
    ] if os.environ.get(k) is not None},
    "env_vars": {k: os.environ.get(k) for k in [
        "CUDA_VISIBLE_DEVICES", "HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME"
    ] if os.environ.get(k) is not None},
    "mixed_precision_note": (
        "This report captures environment capabilities only. Whether mixed precision "
        "(autocast/FP16/BF16) is used must be confirmed from the training configuration/code."
    ),
}

# Torch-level GPU details (if available)
try:
    import torch
    report["torch"] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "bf16_supported": bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)()) if torch.cuda.is_available() else False,
    }
    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": p.name,
                "total_memory_GB": round(p.total_memory / (1024**3), 2),
                "capability": f"{p.major}.{p.minor}",
            })
        report["torch"]["gpus"] = gpus
except Exception as e:
    report["torch"] = f"<torch details unavailable: {e}>"

# Write files
json_path = os.path.join(OUT_DIR, "env_report.json")
txt_path  = os.path.join(OUT_DIR, "env_report.txt")
freeze_path = os.path.join(OUT_DIR, "pip_freeze.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

def to_text(d: dict) -> str:
    lines = []
    lines.append(f"timestamp_utc: {d.get('timestamp_utc')}")
    lines.append(f"python: {d.get('python')}")
    lines.append("")
    lines.append("== PLATFORM ==")
    for k, v in d.get("platform", {}).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("== OS ==")
    lines.append(d.get("os_release", ""))
    lines.append("")
    lines.append("== KERNEL ==")
    lines.append(d.get("kernel", ""))
    lines.append("")
    lines.append("== HARDWARE ==")
    lines.append("-- lscpu --")
    lines.append(d.get("hardware", {}).get("cpu_lscpu", ""))
    lines.append("")
    lines.append("-- free -h --")
    lines.append(d.get("hardware", {}).get("ram_free_h", ""))
    lines.append("")
    lines.append("-- nvidia-smi --")
    lines.append(d.get("hardware", {}).get("gpu_nvidia_smi", ""))
    lines.append("")
    lines.append("-- nvidia-smi (query) --")
    lines.append(d.get("hardware", {}).get("gpu_nvidia_smi_query", ""))
    lines.append("")
    lines.append("== PACKAGES ==")
    for k, v in d.get("packages", {}).items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("== TORCH DETAILS ==")
    lines.append(str(d.get("torch", "")))
    lines.append("")
    lines.append("== WANDB ENV ==")
    lines.append(str(d.get("wandb_env", {})))
    lines.append("")
    lines.append("== RELEVANT ENV VARS ==")
    lines.append(str(d.get("env_vars", {})))
    lines.append("")
    lines.append("== MIXED PRECISION NOTE ==")
    lines.append(d.get("mixed_precision_note", ""))
    return "\n".join(lines)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(to_text(report))

freeze = run(f"{sys.executable} -m pip freeze")
with open(freeze_path, "w", encoding="utf-8") as f:
    f.write(freeze + "\n")

print("Wrote:")
print(" -", json_path)
print(" -", txt_path)
print(" -", freeze_path)
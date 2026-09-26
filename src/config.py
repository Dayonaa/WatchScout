"""
src/config.py - Konfigurasi Terpusat untuk Watch Dealer RAG & NER Engine.
Membaca variabel lingkungan (.env) untuk path database, model, dynamic rules, dan dataset.
Mendukung standalone deployment out-of-the-box dengan default path lokal.
"""

import os
from pathlib import Path

# Root proyek (2 tingkat di atas file ini: src/config.py -> root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File .env
ENV_FILE = PROJECT_ROOT / ".env"


def load_env_file(env_path: Path = ENV_FILE) -> None:
    """Membaca file .env jika ada dan mengisi os.environ tanpa menimpa environment yang sudah ada."""
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as e:
        print(f"⚠️ [Config] Gagal membaca .env: {e}")


# Muat variabel environment dari .env
load_env_file()

# Path Konfigurasi (dapat dioverride via variabel lingkungan di .env)
DB_PATH = Path(os.getenv("WATCH_DB_PATH", PROJECT_ROOT / "data" / "watches.duckdb")).resolve()
MODEL_DIR = Path(os.getenv("WATCH_MODEL_DIR", PROJECT_ROOT / "models" / "watch_ner")).resolve()
RULES_PATH = Path(os.getenv("WATCH_RULES_PATH", PROJECT_ROOT / "models" / "dynamic_rules.json")).resolve()
DATASETS_DIR = Path(os.getenv("WATCH_DATASETS_DIR", PROJECT_ROOT / "datasets")).resolve()

# Turunan path dataset
UNPARSED_PATH = Path(os.getenv("WATCH_UNPARSED_PATH", DATASETS_DIR / "unparsed_candidates.jsonl")).resolve()
TRAIN_DATA_PATH = Path(os.getenv("WATCH_TRAIN_DATA_PATH", DATASETS_DIR / "train_ner.jsonl")).resolve()
BROADCASTS_CSV_PATH = Path(os.getenv("WATCH_BROADCASTS_CSV_PATH", DATASETS_DIR / "watch_broadcasts.csv")).resolve()
RAW_BROADCASTS_PATH = Path(os.getenv("WATCH_RAW_BROADCASTS_PATH", DATASETS_DIR / "raw_broadcasts.txt")).resolve()

# Gemini GenAI Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def ensure_directories():
    """Memastikan direktori yang dibutuhkan tersedia di sistem."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


ensure_directories()


def get_device_info() -> dict:
    """Mendeteksi akselerasi GPU (CUDA) atau fallback otomatis ke CPU."""
    env_device = os.getenv("WATCH_DEVICE", "auto").lower()

    has_onnx_cuda = False
    try:
        import onnxruntime as ort
        has_onnx_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        pass

    has_torch_cuda = False
    try:
        import torch
        has_torch_cuda = torch.cuda.is_available()
    except Exception:
        pass

    cuda_available = has_onnx_cuda or has_torch_cuda

    if env_device == "cpu":
        active = "cpu"
    elif env_device == "cuda":
        active = "cuda"
    else:
        active = "cuda" if cuda_available else "cpu"

    return {
        "active_device": active,
        "cuda_available": cuda_available,
        "onnx_cuda": has_onnx_cuda,
        "torch_cuda": has_torch_cuda,
    }


DEVICE_INFO = get_device_info()
DEVICE = DEVICE_INFO["active_device"]

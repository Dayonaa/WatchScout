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
            val = val.strip()
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            else:
                if "#" in val:
                    val = val.split("#", 1)[0].strip()
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

# Remote PostgreSQL Configuration
DB_CONNECTION = os.getenv("DB_CONNECTION", "pgsql")
DB_HOST = os.getenv("DB_HOST", "34.21.152.179")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_DATABASE = os.getenv("DB_DATABASE", "scraper")
DB_USERNAME = os.getenv("DB_USERNAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_TABLE = os.getenv("DB_TABLE", "watch_broadcasts")

# Sync & Polling Config
SYNC_STATE_PATH = Path(os.getenv("WATCH_SYNC_STATE_PATH", DATASETS_DIR / "postgres_sync_state.json")).resolve()
SYNC_INTERVAL = float(os.getenv("WATCH_SYNC_INTERVAL", "5.0"))
SYNC_BATCH_SIZE = int(os.getenv("WATCH_SYNC_BATCH_SIZE", "100"))

# Gemini GenAI Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Parameter Pelatihan, Retraining & Ekspansi LLM (bisa disetel via .env)
TRAIN_ITER = int(os.getenv("WATCH_TRAIN_ITER", "20"))
AUTO_TRAIN_EVERY = int(os.getenv("WATCH_AUTO_TRAIN_EVERY", "25"))
AUTO_EXPAND_EVERY = int(os.getenv("WATCH_AUTO_EXPAND_EVERY", "10"))
STREAM_SPEED = float(os.getenv("WATCH_STREAM_SPEED", "0.2"))
EXPAND_AUTO_RETRAIN = os.getenv("WATCH_EXPAND_RETRAIN", "true").lower() in ("true", "1", "yes")

_raw_expand_max = int(os.getenv("WATCH_EXPAND_MAX", "0"))
EXPAND_MAX = _raw_expand_max if _raw_expand_max > 0 else None


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

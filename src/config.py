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

def _clean_env(val: str | None, default: str = "") -> str:
    if val is None:
        return default
    # Hapus inline comment jika ada (# ...)
    if "#" in val:
        val = val.split("#", 1)[0]
    return val.strip().strip("'\"")


def _get_str(key: str, default: str) -> str:
    return _clean_env(os.getenv(key), default)


def _get_int(key: str, default: int) -> int:
    raw = _clean_env(os.getenv(key), str(default))
    try:
        return int(raw)
    except Exception:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _clean_env(os.getenv(key), str(default))
    try:
        return float(raw)
    except Exception:
        return default


def _get_bool(key: str, default: bool) -> bool:
    raw = _clean_env(os.getenv(key), str(default)).lower()
    return raw in ("true", "1", "yes")


# Remote PostgreSQL Configuration
DB_CONNECTION = _get_str("DB_CONNECTION", "pgsql")
DB_HOST = _get_str("DB_HOST", "34.21.152.179")
DB_PORT = _get_int("DB_PORT", 5432)
DB_DATABASE = _get_str("DB_DATABASE", "scraper")
DB_USERNAME = _get_str("DB_USERNAME", "postgres")
DB_PASSWORD = _get_str("DB_PASSWORD", "")
DB_TABLE = _get_str("DB_TABLE", "watch_broadcasts")

# Sync & Polling Config
SYNC_STATE_PATH = Path(_get_str("WATCH_SYNC_STATE_PATH", str(DATASETS_DIR / "postgres_sync_state.json"))).resolve()
SYNC_INTERVAL = _get_float("WATCH_SYNC_INTERVAL", 5.0)
SYNC_BATCH_SIZE = _get_int("WATCH_SYNC_BATCH_SIZE", 100)

# Gemini GenAI Config
GEMINI_API_KEY = _get_str("GEMINI_API_KEY", "")
GEMINI_MODEL = _get_str("GEMINI_MODEL", "gemini-3.8-flash")

# Parameter Pelatihan, Retraining & Ekspansi LLM (bisa disetel via .env)
TRAIN_ITER = _get_int("WATCH_TRAIN_ITER", 20)
AUTO_TRAIN_EVERY = _get_int("WATCH_AUTO_TRAIN_EVERY", 25)
AUTO_EXPAND_EVERY = _get_int("WATCH_AUTO_EXPAND_EVERY", 10)
STREAM_SPEED = _get_float("WATCH_STREAM_SPEED", 0.2)
EXPAND_AUTO_RETRAIN = _get_bool("WATCH_EXPAND_RETRAIN", True)

_raw_expand_max = _get_int("WATCH_EXPAND_MAX", 0)
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

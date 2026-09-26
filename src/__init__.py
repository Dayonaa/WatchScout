"""
src - Luxury Watch Dealer Broadcast Extraction & RAG Engine Package.
"""

from src.extractor import (
    WatchBroadcastExtractor,
    WatchItem,
    ExtractionReport,
    default_extractor,
)

from src.reference_decoder import (
    decode_reference,
    extract_explicit_dial,
    parse_price,
    split_glued_token,
)

from src.rag_engine import (
    WatchRAGEngine,
    default_rag,
)

from src.trainer import (
    train_model,
    retrain_from_jsonl,
    append_to_training_buffer,
)

from src.llm_pattern_expander import (
    expand_patterns_with_gemini,
)

from src.config import (
    PROJECT_ROOT,
    DB_PATH,
    MODEL_DIR,
    RULES_PATH,
    DATASETS_DIR,
    UNPARSED_PATH,
)

__all__ = [
    "WatchBroadcastExtractor",
    "WatchItem",
    "ExtractionReport",
    "default_extractor",
    "decode_reference",
    "extract_explicit_dial",
    "parse_price",
    "split_glued_token",
    "WatchRAGEngine",
    "default_rag",
    "train_model",
    "retrain_from_jsonl",
    "append_to_training_buffer",
    "expand_patterns_with_gemini",
    "PROJECT_ROOT",
    "DB_PATH",
    "MODEL_DIR",
    "RULES_PATH",
    "DATASETS_DIR",
    "UNPARSED_PATH",
]

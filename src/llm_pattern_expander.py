"""
src/llm_pattern_expander.py - Autonomous LLM Pattern Expander & Rule Generator.
Menggunakan Gemini 3.8 Flash untuk menganalisis format pesan dealer yang belum dikenal,
mengekstrak entitas jam tangan mewah, membuat aturan referensi dinamis (dynamic_rules.json),
dan menyuntikkan data latihan baru ke train_ner.jsonl sebelum spaCy CPU di-retrain.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from src.genai_client import get_gemini_client, DEFAULT_MODEL
from src.extractor import WatchItem
from src.rag_engine import default_rag
from src.trainer import append_to_training_buffer


from src.config import DATASETS_DIR, MODEL_DIR as MODELS_DIR, UNPARSED_PATH, RULES_PATH as DYNAMIC_RULES_PATH


SYSTEM_PROMPT = """You are an expert luxury watch horologist and natural language processing specialist.
Your task is to analyze unparsed luxury watch dealer broadcast lines from WhatsApp (Hong Kong / Global watch markets).

For the provided lines:
1. Determine if a line refers to a genuine luxury watch (Rolex, Patek Philippe, Audemars Piguet, Cartier, Tudor, Vacheron Constantin, Omega, etc.).
2. Extract the structured fields:
   - brand: Brand name (e.g., Rolex, Patek Philippe, Cartier)
   - series: Model series (e.g., Pearlmaster 39, Nautilus Travel Time, Tank Louis)
   - reference: Model reference number in uppercase (e.g., 86409RBR, 5990/1R, 7010/1R)
   - dial: Dial color or special dial (e.g., Purple, Ombre Green, Silver, Black, etc. or null)
   - material: Case material (e.g., 18k White Gold, Rose Gold, Stainless Steel, etc. or null)
   - price_raw: Original price string if present (e.g., hkd1010000, 97000, $48,000 or null)
   - price_num: Numeric price as float (e.g., 1010000.0 or null)
   - currency: Currency (e.g., HKD, USD)
   - year: Production or card year as integer (e.g., 2016, 2024 or null)
   - condition: NEW or USED (slang like N8, N9, new = NEW; used, 2nd = USED)
3. For any recognized reference, extract a 4-to-6 character prefix rule for future automatic matching.

Respond ONLY with valid JSON (no markdown formatting, no backticks, no code fences):
{
  "new_rules": {
    "PREFIX_CODE": {
      "brand": "Brand Name",
      "series": "Series Name",
      "material": "Material Description"
    }
  },
  "extracted_watches": [
    {
      "raw_text": "original line text",
      "brand": "Rolex",
      "series": "Pearlmaster 39",
      "reference": "86409RBR",
      "dial": null,
      "material": "18k White Gold & Diamonds",
      "price_raw": "hkd1010000",
      "price_num": 1010000.0,
      "currency": "HKD",
      "year": 2016,
      "condition": "USED"
    }
  ]
}
"""


def load_dynamic_rules() -> Dict[str, Any]:
    """Memuat aturan referensi dinamis yang dihasilkan Gemini."""
    if not DYNAMIC_RULES_PATH.exists():
        return {"prefix_map": {}, "dial_map": {}}
    try:
        return json.loads(DYNAMIC_RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"prefix_map": {}, "dial_map": {}}


def save_dynamic_rules(new_prefixes: Dict[str, Any]):
    """Menyimpan atau menggabungkan aturan prefix baru ke dynamic_rules.json."""
    current = load_dynamic_rules()
    prefix_map = current.get("prefix_map", {})

    for pfx, info in new_prefixes.items():
        clean_pfx = pfx.upper().strip()
        if clean_pfx:
            prefix_map[clean_pfx] = info

    current["prefix_map"] = prefix_map
    DYNAMIC_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    DYNAMIC_RULES_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def expand_patterns_with_gemini(
    max_lines: int = 15,
    unparsed_path: Path = UNPARSED_PATH
) -> Dict[str, Any]:
    """
    Mengambil baris unparsed dari antrean, meminta Gemini menganalisisnya,
    membuat aturan referensi baru, dan menyuntikkan data latihan baru ke buffer spaCy.
    """
    if not unparsed_path.exists():
        return {"status": "empty", "message": "Tidak ada berkas unparsed_candidates.jsonl"}

    candidates: List[Dict[str, Any]] = []
    with open(unparsed_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    candidates.append(json.loads(line))
                except Exception:
                    pass

    if not candidates:
        return {"status": "empty", "message": "Antrean unparsed kosong"}

    # Urutkan berdasarkan frekuensi kemunculan tertinggi
    candidates.sort(key=lambda x: x.get("frequency", 1), reverse=True)
    batch = candidates[:max_lines]

    lines_text = "\n".join([f"- {c.get('raw_text')}" for c in batch if c.get("raw_text")])

    print(f"🤖 [Gemini 3.8 Flash] Menganalisis {len(batch)} format unparsed teratas...")
    client = get_gemini_client()

    prompt = f"{SYSTEM_PROMPT}\n\nUnparsed dealer lines to analyze:\n{lines_text}"
    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt
        )
        raw_text = response.text or ""

        # Bersihkan pembungkus markdown jika ada
        clean_json_str = raw_text.strip()
        if clean_json_str.startswith("```"):
            clean_json_str = re.sub(r"^```(?:json)?\n", "", clean_json_str)
            clean_json_str = re.sub(r"\n```$", "", clean_json_str)

        data = json.loads(clean_json_str)
    except Exception as e:
        return {"status": "error", "message": f"Gagal memanggil atau mem-parse output Gemini: {e}"}

    new_rules = data.get("new_rules", {})
    extracted_watches = data.get("extracted_watches", [])

    # 1. Simpan aturan referensi baru ke dynamic_rules.json
    if new_rules:
        save_dynamic_rules(new_rules)
        print(f"✨ [Aturan Baru] Berhasil mendaftarkan {len(new_rules)} pola referensi baru ke dynamic_rules.json:")
        for pfx, info in new_rules.items():
            print(f"   • Prefix '{pfx}': {info.get('brand')} {info.get('series')} ({info.get('material')})")

    # 2. Konversi hasil Gemini menjadi WatchItem dan tambahkan ke Active Learning Buffer
    valid_items: List[WatchItem] = []
    for w in extracted_watches:
        if not w.get("reference"):
            continue
        item = WatchItem(
            brand=w.get("brand") or "Unknown Brand",
            series=w.get("series"),
            reference=w.get("reference"),
            dial=w.get("dial"),
            material=w.get("material"),
            year=w.get("year"),
            condition=w.get("condition") or "USED",
            currency=w.get("currency") or "HKD",
            price_raw=w.get("price_raw"),
            price_num=float(w["price_num"]) if w.get("price_num") else None,
            raw_text=w.get("raw_text") or w.get("reference")
        )
        valid_items.append(item)

    added_to_buffer = 0
    if valid_items:
        # Masukkan ke buffer training spaCy
        added_to_buffer = append_to_training_buffer(valid_items)
        # Indeks juga ke RAG DuckDB
        default_rag.index_items(valid_items)
        print(f"📚 [Active Learning] +{added_to_buffer} contoh latihan baru disuntikkan ke train_ner.jsonl")

    # 3. Hapus seluruh baris yang sudah dianalisis dalam batch dari antrean (baik jam valid maupun teks non-jam/obrolan)
    processed_texts = {c.get("raw_text", "").strip() for c in batch if c.get("raw_text")}
    remaining = [c for c in candidates if c.get("raw_text", "").strip() not in processed_texts]

    with open(unparsed_path, "w", encoding="utf-8") as f:
        for c in remaining:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    discarded_noise = len(processed_texts) - len(valid_items)
    print(f"🧹 [Antrean Dibersihkan] {len(processed_texts)} baris diproses ({len(valid_items)} jam valid, {discarded_noise} baris noise/bukan jam dibuang).")
    print(f"   • Sisa antrean unparsed: {len(remaining)} baris")

    return {
        "status": "success",
        "new_rules_count": len(new_rules),
        "extracted_watches_count": len(valid_items),
        "added_to_buffer": added_to_buffer,
        "remaining_unparsed": len(remaining)
    }

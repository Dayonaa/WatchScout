"""
src/trainer.py - Script Pelatihan Base Model Custom spaCy NER untuk Dealer Broadcast.
Mengonversi teks mentah menjadi DocBin dan melatih model transition-based NER di CPU.
"""

import os
import re
import json
import random
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Any

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extractor import WatchBroadcastExtractor, default_extractor


from src.config import (
    DATASETS_DIR,
    MODEL_DIR,
    TRAIN_DATA_PATH,
    RAW_BROADCASTS_PATH,
)
MODELS_DIR = MODEL_DIR.parent


def generate_training_data(raw_text_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Mengubah broadcast mentah menjadi data anotasi berspan karakter untuk training spaCy NER.
    Label yang dilatih: REFERENCE, YEAR, PRICE, CONDITION, DIAL.
    """
    text = raw_text_path.read_text(encoding="utf-8")
    extractor = WatchBroadcastExtractor()
    training_data = []

    for line in text.splitlines():
        line = line.strip()
        if not line or re.match(r"^[_=\-*\s🔥🇭🇰]+$", line):
            continue

        item = extractor.extract_line(line)
        if not item:
            continue

        entities = []

        # 1. Cari span REFERENCE
        ref_match = re.search(re.escape(item.reference), line, re.IGNORECASE)
        if ref_match:
            entities.append((ref_match.start(), ref_match.end(), "REFERENCE"))

        # 2. Cari span YEAR
        if item.year:
            year_match = re.search(r"\b" + str(item.year) + r"\b", line)
            if year_match:
                entities.append((year_match.start(), year_match.end(), "YEAR"))

        # 3. Cari span PRICE
        if item.price_raw:
            price_match = re.search(re.escape(item.price_raw), line, re.IGNORECASE)
            if price_match:
                entities.append((price_match.start(), price_match.end(), "PRICE"))

        # 4. Cari span CONDITION jika eksplisit di baris
        for cond_word in ["NEW", "BNIB", "USED", "UNWORN", "LIKE NEW"]:
            cond_m = re.search(r"\b" + cond_word + r"\b", line, re.IGNORECASE)
            if cond_m:
                entities.append((cond_m.start(), cond_m.end(), "CONDITION"))
                break

        # Filter entitas yang tumpang tindih (overlapping)
        entities = sorted(entities, key=lambda x: x[0])
        clean_entities = []
        last_end = -1
        for start, end, label in entities:
            if start >= last_end:
                clean_entities.append((start, end, label))
                last_end = end

        if clean_entities:
            training_data.append((line, {"entities": clean_entities}))

    return training_data


def train_model(
    data: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path = MODEL_DIR,
    n_iter: int = 30
):
    """
    Melatih model spaCy NER kustom dari nol di CPU.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "meta.json"

    # 1. Baca metadata sebelumnya untuk auto-increment versi & histori
    prev_version = "0.1.0"
    total_runs = 0
    if meta_path.exists():
        try:
            old_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            raw_ver = old_meta.get("version", "0.1.0")
            total_runs = old_meta.get("total_train_runs", 0)
            parts = raw_ver.split(".")
            if len(parts) == 3 and raw_ver.startswith("0.1."):
                parts[-1] = str(int(parts[-1]) + 1)
                prev_version = ".".join(parts)
            else:
                prev_version = "0.1.0"
        except Exception:
            pass

    # 2. Hitung statistik distribusi entitas
    entity_counts: Dict[str, int] = {}
    for _, annotations in data:
        for ent in annotations.get("entities", []):
            label = ent[2]
            entity_counts[label] = entity_counts.get(label, 0) + 1

    # Buat blank spaCy pipeline bahasa Inggris/Universal
    nlp = spacy.blank("en")

    # Tambahkan komponen NER
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Daftarkan semua label entitas
    for label in entity_counts.keys():
        ner.add_label(label)

    # Siapkan data training dengan alignment span token yang rapi (bebas warning W030)
    from spacy.util import filter_spans
    train_examples = []
    for text, annotations in data:
        doc = nlp.make_doc(text)
        valid_spans = []
        for start, end, label in annotations.get("entities", []):
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
            if span is not None:
                valid_spans.append(span)

        clean_spans = filter_spans(valid_spans)
        example = Example.from_dict(
            doc,
            {"entities": [(s.start_char, s.end_char, s.label_) for s in clean_spans]}
        )
        train_examples.append(example)

    # 0. Hardware acceleration auto-detect (GPU CUDA vs CPU)
    has_gpu = False
    try:
        has_gpu = spacy.prefer_gpu()
    except Exception:
        pass
    hw_label = "CUDA GPU" if has_gpu else "CPU"

    print(f"🚀 Training {len(train_examples)} data (v{prev_version}) di {hw_label}:")
    optimizer = nlp.begin_training()

    # Training loop dengan single-line tqdm progress bar
    from tqdm import tqdm
    batch_sizes = compounding(8.0, 64.0, 1.001) if has_gpu else compounding(4.0, 32.0, 1.001)
    final_loss = 0.0

    with tqdm(total=n_iter, desc=f"🧠 [spaCy {hw_label} v{prev_version}]", unit="iter", ncols=80, leave=True) as pbar:
        for itn in range(n_iter):
            random.shuffle(train_examples)
            losses = {}
            batches = minibatch(train_examples, size=batch_sizes)
            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses, sgd=optimizer)

            final_loss = losses.get("ner", 0.0)
            pbar.set_postfix({"loss": f"{final_loss:.4f}"})
            pbar.update(1)

    # Simpan model hasil training ke disk
    nlp.to_disk(output_dir)

    # 3. Hitung sidik jari (checksum SHA256) bobot neural network (ner/model)
    weight_file = output_dir / "ner" / "model"
    weight_sha = "N/A"
    weight_size_kb = 0.0
    if weight_file.exists():
        raw_weights = weight_file.read_bytes()
        weight_sha = hashlib.sha256(raw_weights).hexdigest()[:16]
        weight_size_kb = len(raw_weights) / 1024

    # 4. Catat record training ke meta.json
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_run_stats = {
        "version": prev_version,
        "timestamp": now_str,
        "dataset_examples": len(train_examples),
        "iterations": n_iter,
        "final_loss": round(float(final_loss), 6),
        "weight_fingerprint": weight_sha,
        "weight_size_kb": round(weight_size_kb, 2),
        "entity_distribution": entity_counts
    }
    # Simpan riwayat lengkap ke file append-only terpisah agar meta.json tetap ringkas
    history_file = output_dir / "history.jsonl"
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(current_run_stats, ensure_ascii=False) + "\n")

    # Baca ulang meta.json yang digenerate oleh spacy lalu simpan versi ringkas
    meta_data = json.loads(meta_path.read_text(encoding="utf-8"))
    meta_data["name"] = "watch_dealer_ner"
    meta_data["version"] = prev_version
    meta_data["description"] = "Custom CPU/GPU optimized spaCy NER for luxury watch dealer broadcasts"
    meta_data["total_train_runs"] = total_runs + 1
    meta_data["latest_training"] = current_run_stats
    # Buang riwayat array panjang dari meta.json agar file tidak membengkak
    meta_data.pop("training_history", None)

    meta_path.write_text(json.dumps(meta_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Model berhasil dilatih & disimpan ke: {output_dir.resolve()}")
    print("=" * 60)
    print(f"  • Versi Model      : v{prev_version}")
    print(f"  • Weight Fingerprint : {weight_sha} ({weight_size_kb:.1f} KB)")
    print(f"  • Final Loss       : {final_loss:.6f}")
    print(f"  • Total Anotasi    : {sum(entity_counts.values())} entitas ({entity_counts})")
    print(f"  • File Metadata    : {meta_path.resolve()}")
    print("=" * 60 + "\n")

    return nlp


def append_to_training_buffer(items: List[Any], buffer_path: Path = TRAIN_DATA_PATH) -> int:
    """
    Menambahkan WatchItem yang baru diekstrak dari stream ke buffer active learning.
    Membuat anotasi span karakter (REFERENCE, YEAR, PRICE, CONDITION, DIAL).
    """
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    existing_texts = set()
    if buffer_path.exists():
        try:
            for line in buffer_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    j = json.loads(line)
                    existing_texts.add(j.get("text", "").strip())
        except Exception:
            pass

    added_count = 0
    with open(buffer_path, "a", encoding="utf-8") as f:
        for it in items:
            line_text = it.raw_text.strip()
            if not line_text or line_text in existing_texts:
                continue

            entities = []
            # Span reference
            if it.reference:
                # Coba cari nomor referensi dasar
                clean_ref = it.reference.split("/")[0]
                m_ref = re.search(re.escape(it.reference), line_text, re.IGNORECASE)
                if not m_ref and clean_ref:
                    m_ref = re.search(re.escape(clean_ref), line_text, re.IGNORECASE)
                if m_ref:
                    entities.append((m_ref.start(), m_ref.end(), "REFERENCE"))

            # Span year
            if it.year:
                m_yr = re.search(r"\b" + str(it.year) + r"\b", line_text)
                if m_yr:
                    entities.append((m_yr.start(), m_yr.end(), "YEAR"))

            # Span price
            if it.price_raw:
                m_pr = re.search(re.escape(it.price_raw), line_text, re.IGNORECASE)
                if m_pr:
                    entities.append((m_pr.start(), m_pr.end(), "PRICE"))

            # Span dial
            if it.dial:
                m_dl = re.search(re.escape(it.dial), line_text, re.IGNORECASE)
                if m_dl:
                    entities.append((m_dl.start(), m_dl.end(), "DIAL"))

            # Urutkan dan buang overlap
            entities = sorted(entities, key=lambda x: x[0])
            clean_entities = []
            last_end = -1
            for start, end, label in entities:
                if start >= last_end:
                    clean_entities.append((start, end, label))
                    last_end = end

            if clean_entities:
                record = {"text": line_text, "entities": clean_entities}
                f.write(json.dumps(record) + "\n")
                existing_texts.add(line_text)
                added_count += 1

    return added_count


def retrain_from_jsonl(
    jsonl_path: Path = TRAIN_DATA_PATH,
    output_dir: Path = MODEL_DIR,
    n_iter: int = 20
) -> Any:
    """
    Melatih ulang model spaCy NER dari seluruh data yang ada di file buffer JSONL.
    """
    if not jsonl_path.exists():
        print(f"File buffer {jsonl_path} belum ada.")
        return None

    data: List[Tuple[str, Dict[str, Any]]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                data.append((j["text"], {"entities": [tuple(e) for e in j["entities"]]}))

    if not data:
        print("Buffer training kosong, pelatihan dibatalkan.")
        return None

    print(f"\n🔄 [Active Learning] Memulai auto-retrain dari {len(data)} sample buffer...")
    return train_model(data, output_dir=output_dir, n_iter=n_iter)


def export_jsonl(data: List[Tuple[str, Dict[str, Any]]], out_path: Path):
    """Mengekspor data training ke format JSONL standar di folder datasets."""
    with open(out_path, "w", encoding="utf-8") as f:
        for text, annot in data:
            f.write(json.dumps({"text": text, "entities": annot["entities"]}) + "\n")
    print(f"📁 Dataset JSONL tersimpan: {out_path.resolve()} ({len(data)} baris)")


def main():
    raw_path = RAW_BROADCASTS_PATH
    if not raw_path.exists():
        print(f"Error: {raw_path} tidak ditemukan!")
        return

    print("1. Menyiapkan data training dari datasets/raw_broadcasts.txt...")
    train_data = generate_training_data(raw_path)

    jsonl_path = TRAIN_DATA_PATH
    export_jsonl(train_data, jsonl_path)

    print("\n2. Melatih model spaCy NER...")
    train_model(train_data, n_iter=25)


if __name__ == "__main__":
    main()

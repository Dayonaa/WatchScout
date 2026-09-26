# ⌚ Watch Dealer NER & Fast RAG Search Engine

Base model ekstraksi informasi entitas jam tangan mewah (*Rolex, Vacheron Constantin, Cartier, Patek Philippe, Audemars Piguet, Tudor*) dan mesin pencarian semantik (RAG) berbasis CPU ultra-cepat.

---

## ⚡ Fitur Utama
* **100% Standalone & Offline**: Berjalan murni di CPU dengan latensi sub-milidetik (< 1 ms per baris) tanpa GPU dan tanpa ketergantungan API eksternal untuk inferensi rutin.
* **Ekstraksi Entitas Menyeluruh**: Mendeteksi Brand, Series, Reference, Dial (warna), Material, Tahun, Kondisi (NEW/USED), dan Harga.
* **Dual-Condition Resolver**: Mendukung penentuan kondisi 2 lapis (header context inheritance + inline override kartu dealer N7/N8/N9).
* **Dealer Traceability**: Menyimpan nomor WhatsApp dealer (`sender_phone`), nama toko (`dealer_alias`), grup asal, dan timestamp di DuckDB untuk transaksi instan.
* **Continuous Active Learning**: Melatih ulang model spaCy NER secara otomatis di background dengan *in-place single-line progress bar* `tqdm` saat data baru terakumulasi.
* **Unparsed Review Queue**: Menampung format pesan yang belum dikenali ke `datasets/unparsed_candidates.jsonl` untuk analisis penemuan pola regex baru.
* **Instant Hybrid RAG**: DuckDB SQL + FastEmbed vector cosine similarity (< 100 ms per pencarian).

---

## 📂 Struktur Direktori Proyek

```text
├── data/                           # Direktori database operasional
│   └── watches.duckdb              # Database vektor lokal DuckDB (terpisah dari datasets)
│
├── datasets/                       # Data broadcast & dataset latih
│   ├── watch_broadcasts.csv        # 1.000.000+ baris chat asli WhatsApp dealer
│   ├── train_ner.jsonl             # Buffer dataset training aktif
│   └── unparsed_candidates.jsonl   # Antrean format baru untuk audit regex
│
├── models/                         # Bobot model & aturan inferensi
│   ├── dynamic_rules.json          # Kamus aturan dinamis hasil ekstensi Gemini
│   └── watch_ner/                  # Bobot model spaCy NER kustom
│       ├── meta.json               # Versi model, loss, dan SHA256 weight fingerprint
│       └── ner/model               # Binary weights neural network transition-based
│
├── src/                            # Modul inti aplikasi (Package)
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Konfigurasi terpusat & path .env
│   ├── extractor.py                # High-speed entity extractor & condition resolver
│   ├── reference_decoder.py        # Kamus domain, dial decoding, dan normalisasi harga
│   ├── rag_engine.py               # DuckDB hybrid vector search & contact metadata
│   ├── trainer.py                  # CPU training loop spaCy NER & token aligner
│   ├── genai_client.py             # Klien resmi Google GenAI (Gemini 3.8 Flash)
│   └── llm_pattern_expander.py     # Otomasi self-expansion pola & aturan regex
│
├── app.py                          # CLI pencarian RAG & shell interaktif
├── stream_simulator.py             # Simulator chat WhatsApp masuk real-time
├── main.py                         # Unified CLI hub terpusat
├── .env.example                    # Template konfigurasi environment
├── pyproject.toml                  # Konfigurasi instalasi mandiri via pip
└── README.md                       # Dokumentasi resmi proyek
```

---

## 📦 Instalasi Mandiri

```bash
# 1. Masuk ke direktori proyek
cd AI

# 2. Aktifkan virtual environment
source .venv/bin/activate

# 3. Install package dan dependencies
pip install -e .

# 4. Setup file .env (opsional jika ingin custom path atau menggunakan fitur Gemini)
cp .env.example .env
# Edit .env dan isi GEMINI_API_KEY Anda
```

### ⚙️ Konfigurasi Jalur / Path (`.env`)
Sistem berjalan 100% standalone out-of-the-box tanpa konfigurasi tambahan. Namun untuk kebutuhan deployment server, Docker, atau shared volume, Anda dapat meng-override opsi di `.env`:
* `WATCH_DEVICE`: Mode akselerasi hardware (`auto`, `cuda`, atau `cpu`)
* `WATCH_DB_PATH`: Lokasi file database DuckDB (default: `data/watches.duckdb`)
* `WATCH_MODEL_DIR`: Lokasi direktori bobot model spaCy NER (default: `models/watch_ner`)
* `WATCH_RULES_PATH`: Lokasi kamus aturan dinamis (default: `models/dynamic_rules.json`)
* `WATCH_DATASETS_DIR`: Lokasi folder dataset (default: `datasets/`)
* `GEMINI_API_KEY`: API Key Google Gemini (opsional, untuk modul `expand`)

### ⚡ Akselerasi GPU NVIDIA (Opsional untuk Server / RTX 3060)
Sistem memiliki fitur **Auto-Detect GPU**. Di server dengan kartu grafis NVIDIA (seperti RTX 3060 12GB), Anda cukup memasang paket akselerasi CUDA berikut agar bulk embedding 1 juta data berjalan super cepat:
```bash
# Pasang ONNX Runtime GPU (CUDA Provider) untuk FastEmbed
pip install onnxruntime-gpu

# Pasang akselerasi CUDA spaCy (opsional untuk training ribuan entitas)
pip install "spacy[cuda12x]"
```
*(Jika paket GPU belum terpasang atau di laptop biasa, sistem otomatis berjalan di mode CPU dengan aman tanpa error).*

---

## 🚀 Panduan Penggunaan

### 1. Dashboard Utama & Status Sistem (`main.py`)
Melihat status database DuckDB, versi model spaCy aktif, dan daftar perintah:
```bash
python main.py
```

### 2. Pencarian Jam & Kontak WhatsApp Dealer
Cari jam tangan secara semantik di database RAG lokal:
```bash
# Cari Rolex dial hijau dengan harga di bawah 85k HKD
python main.py search "126300 ombre green" --max-price 85000

# Shell pencarian interaktif
python main.py search
```

### 3. Simulasi Aliran Chat WhatsApp Dealer Masuk
Simulasi membaca pesan dealer satu per satu dari CSV layaknya pesan live WhatsApp masuk:
```bash
# Jalankan simulasi 30 pesan dengan auto-retrain setiap 25 jam
python main.py stream --limit 30 --speed 0.2 --auto-train-every 25

# Tampilkan seluruh baris jam tanpa pemotongan
python main.py stream --limit 10 --verbose
```

### 4. Tinjau Format Belum Dikenal (Audit Regex Baru)
Lihat variasi chat dealer yang belum berhasil diekstrak untuk bahan penambahan regex baru:
```bash
python main.py review
```

### 5. Melatih Ulang Base Model spaCy NER
Melatih ulang model spaCy dari seluruh buffer data yang terkumpul:
```bash
python main.py train --iter 15
```

---

## 💻 Menggunakan di Kode Python Mandiri

```python
from src import default_extractor, default_rag

# 1. Ekstraksi 1 baris teks dealer
item = default_extractor.extract_line("126300ombre green hkd82500 n8")
print(item.brand)      # Rolex
print(item.series)     # Datejust 41
print(item.reference)  # 126300
print(item.dial)       # Ombre Green
print(item.price_num)  # 82500.0
print(item.condition)  # NEW

# 2. Pencarian RAG
results = default_rag.search("Cartier Panthere gold", limit=3)
for r in results:
    print(r["reference"], r["price"], r["dealer_alias"], r["sender_phone"])
```

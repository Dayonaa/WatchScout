# 📋 Rencana Pengembangan Base Model & RAG: Dealer Broadcast Watch Extractor

Dokumen perencanaan arsitektur, skema entitas (termasuk Dial & Material), teknologi CPU-optimized, simulasi stream real-time, dan pipeline pembelajaran otomatis (*Continuous Active Learning*).

---

## 🎯 1. Tujuan & Sasaran Utama

1. **Inferensi Super Kencang di CPU**: Latensi sub-milidetik (< 1 ms per baris broadcast), mampu memproses ribuan baris teks broadcast per detik tanpa memerlukan GPU.
2. **Ekstraksi Entitas Menyeluruh**:
   * Menangani format pesan dealer yang tidak beraturan, emoji (🇭🇰, 🔥), singkatan harga (`182k`, `hkd 265k`, `hld219000`), kondisi (`NEW`, `Used`), dan nomor referensi yang kompleks.
   * Mendukung ekstraksi **DIAL (Warna/Jenis Dial)** baik dari teks eksplisit maupun decoding kode referensi.
   * Menangani kata yang menempel tanpa spasi (*glued tokens* seperti `126300ombre green` atau `hld219000`).
3. **Simulasi Aliran Chat Real-Time (*Streaming Ingestion*)**:
   * Membaca ribuan chat dealer dari `datasets/watch_broadcasts.csv` (atau JSON) secara bertahap (simulasi pesan WhatsApp masuk satu per satu).
   * Menampilkan visualisasi live di terminal saat AI memproses chat secara real-time.
4. **Pembelajaran Mandiri Otomatis (*Continuous Active Learning*)**:
   * Otomatis mengumpulkan data baru hasil ekstraksi ke dalam buffer training (`datasets/train_ner.jsonl`).
   * Setiap mencapai ambang batas pesan tertentu (misal per 50 atau 100 chat), sistem otomatis melatih ulang (*auto-retrain*) model spaCy di background, memperbarui bobot, dan menaikkan versi di `meta.json`.
5. **RAG & Hybrid Search Cepat**: Pencarian semantik dan terstruktur instan (misal: *"Cari Rolex 126300 dial green tahun 2023 ke atas budget di bawah 85k HKD"*).
6. **100% Standalone & Portable**: Bisa diinstal langsung via `pip install -e .` dan folder bobot model `models/watch_ner` + database `watches.duckdb` bisa dipindahkan ke mesin manapun secara plug-and-play.

---

## 🏷️ 2. Taksonomi & Skema Entitas

### A. Data Jam Tangan (Spesifikasi Produk)
| Entitas | Deskripsi | Contoh Input Teks | Hasil Normalisasi |
| :--- | :--- | :--- | :--- |
| **`BRAND`** | Merek jam tangan | `VC`, `Rolex`, `Cartier`, `AP`, `Tudor` | `Rolex` / `Cartier` / `Vacheron Constantin` |
| **`SERIES`** | Lini / Seri model | `Datejust`, `Overseas`, `Submariner`, `Panthère` | `Datejust 41` / `Sky-Dweller` |
| **`REFERENCE`** | Nomor referensi model | `126300`, `4500v/110a-b483`, `WJPN0085`, `336934` | `126300` / `4500V/110A-B483` |
| **`DIAL`** | Warna / tipe dial | `Ombre Green`, `Blue`, `Silver`, `Black`, `B128` | `Green Dial` / `Blue Dial` |
| **`MATERIAL`** | Bahan case / bezel | `Oystersteel`, `Rose Gold`, `White Gold` | `Stainless Steel` / `Rose Gold` |
| **`YEAR`** | Tahun produksi / kartu | `2024`, `2011`, `2025` | `2024` *(integer)* |
| **`CONDITION`** | Kondisi jam | `Used`, `NEW`, `n7/n8/n9`, `BNIB`, `Unworn` | `USED` / `NEW` |
| **`PRICE_RAW`** | Teks harga asli | `hkd82500`, `HKD182K`, `hld219000` | `HKD 82,500` / `HKD 219,000` |
| **`CURRENCY`** | Mata uang | `HKD`, `USD`, `SGD`, `IDR`, `EUR` | `HKD` |
| **`PRICE_NUM`** | Nilai angka terstandarisasi | `hkd82500` ➔ `82500`, `182K` ➔ `182000` | `82500` *(integer)* |

### B. Metadata Pengirim & Pelacakan Dealer (Traceability di DuckDB)
Menyimpan identitas lengkap dari pengirim WhatsApp agar hasil pencarian RAG langsung dapat ditindaklanjuti untuk transaksi direct contact:

| Kolom DuckDB | Tipe Data | Asal Kolom CSV | Tujuan & Manfaat |
| :--- | :--- | :--- | :--- |
| **`message_id`** | `VARCHAR PRIMARY KEY` | `message_id` / `id` | ID unik pesan untuk mencegah duplikasi data broadcast. |
| **`sender_phone`** | `VARCHAR` | `sender_phone` | Nomor WhatsApp dealer (misal: `+85262735929`) untuk *direct WhatsApp chat*. |
| **`sender_name`** | `VARCHAR` | `sender_push_name` | Nama profil kontak WhatsApp pengirim. |
| **`dealer_alias`** | `VARCHAR` | `dealer_alias` | Nama panggilan / identitas toko dealer (misal: *Derek*, *Global Time*). |
| **`chat_name`** | `VARCHAR` | `chat_name` | Grup WhatsApp tempat broadcast diposting. |
| **`broadcast_time`**| `TIMESTAMP` | `timestamp` | Waktu posting pesan untuk filter stok terbaru (misal: "stok 24 jam terakhir"). |
| **`embedding`** | `FLOAT[]` | Dihitung FastEmbed | Vektor representasi semantik untuk hybrid search. |

---

## 🎨 3. Strategi Khusus Ekstraksi DIAL, Material & Glued Tokens

Informasi Dial dan spesifikasi pada pesan WhatsApp dealer seringkali menempel tanpa spasi (*glued tokens*):

1. **Glued Reference & Dial Words**:
   * Contoh: `126300ombre green` ➔ Regex pemisah: `^(\d{5,6}(?:LN|LV|BLRO|BLNR|CHNR)?)([a-zA-Z\s]+)$`
   * Hasil: Reference = `126300`, Dial = `Ombre Green`
   * Contoh: `336934blue` ➔ Reference = `336934`, Dial = `Blue`
2. **Typo / Glued Currency & Price**:
   * Contoh: `hld219000` ➔ mapping typo `HLD` / `HKD` ➔ Currency = `HKD`, Price = `219,000`
   * Contoh: `hkd82500` ➔ Currency = `HKD`, Price = `82,500`
3. **Pola Dial Berdasarkan Suffix & Kamus**:
   * **Rolex**: Prefix 6 angka (`126300` = Steel, `126334` = Steel/WG Fluted), suffix huruf (`LN` = Black Cerachrom, `LV` = Green Cerachrom, `BLRO` = Pepsi).
   * **Vacheron Constantin**: `B128` (Blue), `B483` (Silver), `B952/B966` (Green), `200R` (Rose Gold), `110A` (Steel).
   * **Cartier**: Awalan `WJPN` (Gold/Diamonds), `WGTA` (Gold), `WSTA` (Steel).

---

## 🛡️ 4. Strategi Penentuan Kondisi (Header vs Inline)

Kondisi jam seringkali ditulis sebagai judul bagian (*header*) atau berdampingan langsung dengan nama produk (*inline*). Masalah ini diatasi menggunakan sistem **Stateful Context Tracking dengan Inline Override (Prioritas 2 Lapis)**:

```
                       [ Baris Pesan Masuk ]
                                 │
                 Apakah baris ini adalah HEADER?
                 (misal: "Used /Used /Used" atau "NEW")
                                 │
                   ┌─────────────┴─────────────┐
                   │ YA                        │ TIDAK (Baris Jam)
                   ▼                           ▼
        Update State Global:         Cek: Apakah baris jam ini punya
        current_condition = USED     kata kunci kondisi sendiri (INLINE)?
                                               │
                                 ┌─────────────┴─────────────┐
                                 │ YA                        │ TIDAK
                                 ▼                           ▼
                           Gunakan INLINE           Gunakan STATE HEADER
                           (Prioritas Tertinggi)    (Inheritance)
```

### Logika Eksekusi di `src/extractor.py`:
1. **Header Detection**:
   * Jika satu baris hanya berisi kata kondisi berulang atau penanda bagian (misal: `Used /Used /Used`, `=== USED STOCKS ===`, `BRAND NEW`), baris tersebut **bukan** produk, melainkan pengubah state:
     `current_condition = "USED"` atau `current_condition = "NEW"`.
2. **Inline Override (Prioritas Tertinggi)**:
   * Jika pada baris produk ditemukan token:
     * `NEW`, `BNIB`, `UNWORN`, atau kode kartu dealer `n7`, `n8`, `n9` (slang dealer Hong Kong untuk *New Card*) ➔ Jam diberi kondisi **`NEW`**.
     * `USED`, `95%`, `98%`, `PREOWNED`, `2nd` ➔ Jam diberi kondisi **`USED`**.
   * Token inline **selalu menimpa** nilai `current_condition` dari header.
3. **Header Inheritance (Prioritas Kedua)**:
   * Jika tidak ada token kondisi inline pada baris produk, jam otomatis mewarisi nilai `current_condition` aktif dari header di atasnya.
   * Default fallback jika tidak ada header maupun inline: `UNKNOWN`.

---

## 🔄 5. Alur Simulasi Stream & Pembelajaran Otomatis (Continuous Learning)

```
[ datasets/watch_broadcasts.csv ] (1.000.000+ baris chat WhatsApp)
                 │
                 ▼  (Loop bertahap dengan delay / simulasi chat masuk)
┌─────────────────────────────────────────────────────────────┐
│ 1. STREAM SIMULATOR (`stream_simulator.py`)                 │
│    • Mengambil baris pesan chat satu per satu               │
│    • Menampilkan tampilan live chat masuk di terminal       │
└──────────────────────────────┬──────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. REAL-TIME AI EXTRACTOR & DECODER                         │
│    • Menjalankan model spaCy NER + Reference Decoder        │
│    • Memisahkan glued tokens & ekstraksi Brand/Ref/Dial/Cond│
│    • Latensi < 1 milidetik di CPU!                          │
└──────────────────────────────┬──────────────────────────────┘
                 │
                 ├──────────────────────────────┐
                 ▼                              ▼
┌──────────────────────────────┐ ┌──────────────────────────────┐
│ 3. DUCKDB & RAG INDEXING     │ │ 4. ACTIVE LEARNING BUFFER    │
│    • Hitung embedding vektor │ │    • Kumpulkan sample baru   │
│    • Simpan spek + kontak    │ │      ke `train_ner.jsonl`    │
│    • Siap dicari instan      │ │    • Counter pesan dipantau  │
└──────────────────────────────┘ └──────────────┬───────────────┘
                                                │
                                                ▼  (Setiap N pesan, misal per 50 chat)
                                 ┌──────────────────────────────┐
                                 │ 5. AUTO-RETRAINING ENGINE    │
                                 │    • Retrain model spaCy     │
                                 │    • Update bobot ner/model  │
                                 │    • Bump version di meta.json│
                                 │    • Catat histori loss baru │
                                 └──────────────────────────────┘
```

---

## 🏗️ 6. Arsitektur Komponen & Teknologi

| Komponen | File / Modul | Peran Utama |
| :--- | :--- | :--- |
| **Stream Simulator** | `stream_simulator.py` | Membaca data CSV/JSON secara streaming (simulasi delay per pesan), menampilkan dashboard progres live di terminal. |
| **Domain Decoder** | `src/reference_decoder.py` | Kamus pola multi-merek (VC, Rolex, Cartier, Tudor, AP, Patek) untuk mengurai Dial, Material, dan Harga. |
| **Base Extractor** | `src/extractor.py` | Parser berkecepatan tinggi (15.000+ baris/detik di CPU) dengan stateful condition tracking dan metadata mapping. |
| **Auto-Trainer** | `src/trainer.py` | Melatih ulang model spaCy NER di CPU, mencatat fingerprint bobot, menghitung penurunan loss, dan memperbarui `meta.json`. |
| **LLM Pattern Expander** | `src/llm_pattern_expander.py` | Agen Gemini 3.8 Flash untuk analisis unparsed lines, pembentukan aturan dinamis `models/dynamic_rules.json`, dan pelabelan data latih baru. |
| **GenAI Client** | `src/genai_client.py` | Manajemen koneksi resmi Google GenAI API dengan integrasi `.env`. |
| **Interactive Shell** | `app.py` | Antarmuka CLI untuk pencarian manual dan indexing berkas. |
| **Packaging Standalone**| `pyproject.toml` | Konfigurasi instalasi mandiri via `pip install -e .`. |

---

## 📋 7. Status Implementasi & Roadmap Progress

- [x] **Arsitektur & Spesifikasi Pipeline**: Selesai dirancang dan didokumentasikan di `PLAN.md`.
- [x] **Base Model spaCy NER & Fingerprint Logger**: Model `v1.0.1` terlatih di CPU dengan pencatatan `meta.json` dan SHA256 checksum.
- [x] **Initial DuckDB Vector Store with FastEmbed**: Engine hybrid RAG tersambung dan diuji dengan model embedding BAAI/bge-small-en-v1.5.
- [x] **Packaging Standalone**: Konfigurasi `pyproject.toml` siap di-install via `pip install -e .`.
- [x] **Multi-Brand Reference Decoder**: Perluasan regex dan kamus untuk Rolex, Cartier, Tudor, Patek Philippe, Audemars Piguet, serta penanganan *glued tokens* (`src/reference_decoder.py`).
- [x] **Dua Lapis Ekstraksi Kondisi & Dealer Metadata**: Implementasi stateful header inheritance + inline override serta penangkapan kontak dealer di `src/extractor.py`.
- [x] **DuckDB Schema Update**: Penambahan kolom `sender_phone`, `dealer_alias`, `chat_name`, `broadcast_time` di tabel `watches` (`src/rag_engine.py`).
- [x] **Stream Simulator CLI (`stream_simulator.py`)**: Script simulasi aliran broadcast dari `datasets/watch_broadcasts.csv` dengan delay terkontrol, log kompak, dan laporan audit kelolosan.
- [x] **Unparsed Review Queue**: Penampungan otomatis format belum dikenal ke `datasets/unparsed_candidates.jsonl` untuk analisis pola baru.
- [x] **LLM-in-the-Loop Pattern Expander**: Gemini 3.8 Flash menganalisis unparsed lines, menghasilkan aturan ke `models/dynamic_rules.json`, dan menyuntikkan data latih ke `train_ner.jsonl` (`src/llm_pattern_expander.py`).
- [x] **Continuous Active Learning Trigger**: Integrasi buffer penampung data baru dan auto-retraining otomatis saat mencapai ambang batas chat (`v1.0.13` terlatih di CPU).
- [x] **Pemisahan Database & Konfigurasi Standalone (.env)**:
  - Relokasi `watches.duckdb` keluar dari `datasets/` menuju `data/watches.duckdb`.
  - Pembuatan modul konfigurasi terpusat `src/config.py` yang membaca path dan credential dari `.env`.
  - Template `.env.example` untuk deployment mandiri dan Docker.
- [x] **Verifikasi & Validasi Akhir**: Pengujian pencarian RAG terpadu menghasilkan data jam lengkap dengan nomor WhatsApp dealer dari lokasi database baru.

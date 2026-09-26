"""
src/rag_engine.py - Mesin RAG & Hybrid Search Cepat Berbasis DuckDB + FastEmbed CPU.
Mendukung pencarian semantik (vektor) dan filter terstruktur (SQL) dengan
penyimpanan metadata lengkap pengirim WhatsApp dealer (traceability & direct contact).
"""

import os
import duckdb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extractor import WatchBroadcastExtractor, WatchItem, default_extractor
from src.config import DB_PATH, DEVICE, DEVICE_INFO


class WatchRAGEngine:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.device = DEVICE
        try:
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            if "Conflicting lock is held" in str(e) or "Could not set lock" in str(e):
                print(f"\n{'='*75}")
                print(f"⚠️  [DUCKDB FILE LOCK CONFLICT] Database Sedang Terkunci!")
                print(f"{'='*75}")
                print(f"File database: {self.db_path}")
                print(f"Penyebab     : Ekstensi IDE (seperti SQLTools / DuckDB Viewer) sedang aktif")
                print(f"               tersambung dan memegang lock eksklusif ke file database.")
                print(f"\n💡 SOLUSI CEPAT (1 KLIK):")
                print(f"  1. Buka sidebar SQLTools / DuckDB di sebelah kiri editor.")
                print(f"  2. Klik tombol/ikon 'Disconnect' (cabut koneksi) pada database.")
                print(f"  3. Jalankan kembali perintah Anda di terminal.")
                print(f"{'='*75}\n")
                sys.exit(1)
            raise e
        self._init_db()

        # Inisialisasi FastEmbed model (Auto-detect GPU CUDA vs CPU)
        if self.device == "cuda":
            try:
                self.embed_model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    cuda=True,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
            except Exception:
                self.device = "cpu"
                self.embed_model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    cuda=False,
                    providers=["CPUExecutionProvider"]
                )
        else:
            self.embed_model = TextEmbedding(
                model_name="BAAI/bge-small-en-v1.5",
                cuda=False,
                providers=["CPUExecutionProvider"]
            )

    def _init_db(self):
        """Membuat tabel watches dengan schema pelacakan dealer jika belum ada."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS watches (
                id VARCHAR PRIMARY KEY,
                brand VARCHAR,
                series VARCHAR,
                reference VARCHAR,
                dial VARCHAR,
                material VARCHAR,
                year INTEGER,
                condition VARCHAR,
                currency VARCHAR,
                price_raw VARCHAR,
                price_num DOUBLE,
                sender_phone VARCHAR,
                sender_name VARCHAR,
                dealer_alias VARCHAR,
                chat_name VARCHAR,
                broadcast_time VARCHAR,
                message_id VARCHAR,
                raw_text VARCHAR,
                search_text VARCHAR,
                embedding FLOAT[]
            )
        """)

        # Migration otomatis jika kolom baru belum ada di DB yang sudah terbentuk
        existing_cols = [c[1] for c in self.conn.execute("PRAGMA table_info('watches')").fetchall()]
        new_cols = {
            "sender_phone": "VARCHAR",
            "sender_name": "VARCHAR",
            "dealer_alias": "VARCHAR",
            "chat_name": "VARCHAR",
            "broadcast_time": "VARCHAR",
            "message_id": "VARCHAR",
        }
        for col_name, col_type in new_cols.items():
            if col_name not in existing_cols:
                try:
                    self.conn.execute(f"ALTER TABLE watches ADD COLUMN {col_name} {col_type}")
                except Exception:
                    pass

    def _make_search_text(self, item: WatchItem) -> str:
        """Membuat teks representasi untuk embedding semantik."""
        parts = [
            item.brand,
            item.series or "",
            item.reference,
            f"{item.dial} dial" if item.dial else "",
            item.material or "",
            str(item.year) if item.year else "",
            item.condition,
            f"{item.currency} {item.price_num:,.0f}" if item.price_num else "",
            item.dealer_alias or ""
        ]
        return " ".join([p for p in parts if p]).strip()

    def index_items(self, items: List[WatchItem]) -> int:
        """
        Memasukkan daftar WatchItem yang sudah diekstrak ke dalam DuckDB + Vector Index.
        """
        if not items:
            return 0

        # Siapkan search text & hitung embedding
        batch_size = 512 if self.device == "cuda" else 64
        embeddings = list(self.embed_model.embed(search_texts, batch_size=batch_size))

        inserted_count = 0
        for it, st, emb in zip(items, search_texts, embeddings):
            # ID unik: reference + message_id/year + hash
            msg_id_part = it.message_id or (it.sender_phone or "UNK")
            doc_id = f"{it.reference}_{it.year or 'NA'}_{abs(hash(it.raw_text + msg_id_part))}"
            emb_list = [float(x) for x in emb]

            self.conn.execute("""
                INSERT OR REPLACE INTO watches (
                    id, brand, series, reference, dial, material, year, condition,
                    currency, price_raw, price_num, sender_phone, sender_name,
                    dealer_alias, chat_name, broadcast_time, message_id,
                    raw_text, search_text, embedding
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                doc_id,
                it.brand,
                it.series,
                it.reference,
                it.dial,
                it.material,
                it.year,
                it.condition,
                it.currency,
                it.price_raw,
                it.price_num,
                it.sender_phone,
                it.sender_name,
                it.dealer_alias,
                it.chat_name,
                it.broadcast_time,
                it.message_id,
                it.raw_text,
                st,
                emb_list
            ))
            inserted_count += 1

        return inserted_count

    def index_broadcast(
        self,
        text: str,
        extractor: Optional[WatchBroadcastExtractor] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Mengekstrak teks broadcast dan memasukkannya ke dalam DuckDB + Vector Index.
        """
        if extractor is None:
            extractor = default_extractor

        items = extractor.extract_broadcast(text, metadata=metadata)
        return self.index_items(items)

    def search(
        self,
        query: str,
        limit: int = 5,
        brand: Optional[str] = None,
        dial: Optional[str] = None,
        max_price: Optional[float] = None,
        min_year: Optional[int] = None,
        condition: Optional[str] = None,
        dealer: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Melakukan pencarian semantik (vektor) dengan filter terstruktur.
        Mengembalikan informasi produk beserta kontak WhatsApp dealer.
        """
        # Hitung embedding query pencarian
        query_emb = list(self.embed_model.embed([query]))[0]
        q_emb_list = [float(x) for x in query_emb]

        # Bangun query SQL dengan cosine similarity
        where_clauses = ["1=1"]
        params: List[Any] = []

        if brand:
            where_clauses.append("LOWER(brand) LIKE ?")
            params.append(f"%{brand.lower()}%")
        if dial:
            where_clauses.append("LOWER(dial) LIKE ?")
            params.append(f"%{dial.lower()}%")
        if max_price:
            where_clauses.append("price_num <= ?")
            params.append(max_price)
        if min_year:
            where_clauses.append("year >= ?")
            params.append(min_year)
        if condition:
            where_clauses.append("condition = ?")
            params.append(condition.upper())
        if dealer:
            where_clauses.append("(LOWER(dealer_alias) LIKE ? OR sender_phone LIKE ?)")
            params.append(f"%{dealer.lower()}%")
            params.append(f"%{dealer}%")

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT 
                brand,
                series,
                reference,
                dial,
                material,
                year,
                condition,
                currency,
                price_num,
                sender_phone,
                sender_name,
                dealer_alias,
                chat_name,
                broadcast_time,
                raw_text,
                list_cosine_similarity(embedding, ?) as similarity
            FROM watches
            WHERE {where_sql}
            ORDER BY similarity DESC
            LIMIT ?
        """
        all_params = [q_emb_list] + params + [limit]
        rows = self.conn.execute(sql, all_params).fetchall()

        results = []
        for r in rows:
            results.append({
                "brand": r[0],
                "series": r[1],
                "reference": r[2],
                "dial": r[3],
                "material": r[4],
                "year": r[5],
                "condition": r[6],
                "currency": r[7],
                "price": r[8],
                "sender_phone": r[9],
                "sender_name": r[10],
                "dealer_alias": r[11],
                "chat_name": r[12],
                "broadcast_time": r[13],
                "raw_text": r[14],
                "score": round(float(r[15]), 4)
            })

        return results

    def count_watches(self) -> int:
        """Mengembalikan total jam yang tersimpan di database."""
        res = self.conn.execute("SELECT COUNT(*) FROM watches").fetchone()
        return res[0] if res else 0


default_rag = WatchRAGEngine()

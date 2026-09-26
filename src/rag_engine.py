"""
src/rag_engine.py - Mesin RAG & Hybrid Search Cepat Berbasis DuckDB + FastEmbed CPU.
Mendukung pencarian semantik (vektor) dan filter terstruktur (SQL) dengan
penyimpanan metadata lengkap pengirim WhatsApp dealer (traceability & direct contact).
"""

import os
import hashlib
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
    def __init__(self, db_path: str = str(DB_PATH), read_only: bool = False, auto_connect: bool = False):
        self.db_path = db_path
        self.read_only = read_only
        self.device = DEVICE
        self._conn = None
        self._embed_model = None
        if auto_connect:
            self._connect()
            if not self.read_only:
                self._init_db()

    @property
    def embed_model(self):
        """Inisialisasi FastEmbed model secara lazy (hanya saat dibutuhkan)."""
        if self._embed_model is None:
            if self.device == "cuda":
                try:
                    self._embed_model = TextEmbedding(
                        model_name="BAAI/bge-small-en-v1.5",
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                except Exception:
                    self.device = "cpu"
                    self._embed_model = TextEmbedding(
                        model_name="BAAI/bge-small-en-v1.5",
                        providers=["CPUExecutionProvider"]
                    )
            else:
                self._embed_model = TextEmbedding(
                    model_name="BAAI/bge-small-en-v1.5",
                    providers=["CPUExecutionProvider"]
                )
        return self._embed_model

    @property
    def conn(self):
        """Auto-connect ke DuckDB saat property conn diakses jika belum terhubung."""
        if self._conn is None:
            self._connect()
        return self._conn

    @conn.setter
    def conn(self, val):
        self._conn = val

    def _connect(self, retries: int = 6, delay: float = 0.5):
        if self._conn is not None:
            return self._conn
        import time
        last_err = None
        for i in range(retries):
            try:
                self._conn = duckdb.connect(self.db_path, read_only=self.read_only)
                return self._conn
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "Conflicting lock is held" in err_str or "Could not set lock" in err_str:
                    time.sleep(delay)
                    continue
                raise e
        if last_err:
            if "Conflicting lock is held" in str(last_err) or "Could not set lock" in str(last_err):
                if self.read_only:
                    import shutil
                    import tempfile
                    try:
                        snap_path = Path(tempfile.gettempdir()) / "watchscout_read_snapshot.duckdb"
                        shutil.copy2(str(self.db_path), str(snap_path))
                        wal_path = Path(str(self.db_path) + ".wal")
                        if wal_path.exists():
                            shutil.copy2(str(wal_path), str(snap_path) + ".wal")
                        self._conn = duckdb.connect(str(snap_path), read_only=True)
                        return self._conn
                    except Exception:
                        pass
                print(f"\n{'='*75}")
                print(f"⚠️  [DUCKDB FILE LOCK CONFLICT] Database Sedang Terkunci!")
                print(f"{'='*75}")
                print(f"File database: {self.db_path}")
                print(f"Penyebab     : Proses lain (atau daemon) sedang memegang lock eksklusif.")
                print(f"Saran        : Tunggu beberapa detik atau gunakan mode read_only.")
                print(f"{'='*75}\n")
            raise last_err

    def close(self):
        """Menutup koneksi DuckDB agar file lock terlepas untuk proses lain."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def _init_db(self):
        """Membuat tabel watches dengan schema pelacakan dealer jika belum ada."""
        self._connect()
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
                media_path VARCHAR,
                media_url VARCHAR,
                thumbnail VARCHAR,
                mimetype VARCHAR,
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
            "media_path": "VARCHAR",
            "media_url": "VARCHAR",
            "thumbnail": "VARCHAR",
            "mimetype": "VARCHAR",
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
        search_texts = [self._make_search_text(it) for it in items]
        batch_size = 512 if self.device == "cuda" else 64
        embeddings = list(self.embed_model.embed(search_texts, batch_size=batch_size))

        self._connect()
        inserted_count = 0
        for it, st, emb in zip(items, search_texts, embeddings):
            # ID unik deterministik (SHA256) untuk menjamin idempotensi de-duplikasi antar proses
            seed = f"{it.reference}_{it.year or 'NA'}_{it.price_num or 'NA'}_{it.condition}_{it.sender_phone or 'UNK'}_{it.raw_text}"
            doc_hash = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
            doc_id = f"{it.reference}_{it.year or 'NA'}_{doc_hash}"
            emb_list = [float(x) for x in emb]

            self.conn.execute("""
                INSERT OR REPLACE INTO watches (
                    id, brand, series, reference, dial, material, year, condition,
                    currency, price_raw, price_num, sender_phone, sender_name,
                    dealer_alias, chat_name, broadcast_time, message_id,
                    media_path, media_url, thumbnail, mimetype,
                    raw_text, search_text, embedding
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                it.media_path,
                it.media_url,
                it.thumbnail,
                it.mimetype,
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
                media_path,
                media_url,
                thumbnail,
                mimetype,
                raw_text,
                list_cosine_similarity(embedding, ?) as similarity
            FROM watches
            WHERE {where_sql}
            ORDER BY similarity DESC
            LIMIT ?
        """
        all_params = [q_emb_list] + params + [limit]
        self._connect()
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
                "media_path": r[14],
                "media_url": r[15],
                "thumbnail": r[16],
                "mimetype": r[17],
                "raw_text": r[18],
                "score": round(float(r[19]), 4)
            })

        return results

    def count_watches(self) -> int:
        """Mengembalikan total jam yang tersimpan di database."""
        self._connect()
        res = self.conn.execute("SELECT COUNT(*) FROM watches").fetchone()
        return res[0] if res else 0


default_rag = WatchRAGEngine()

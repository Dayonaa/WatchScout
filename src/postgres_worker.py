"""
src/postgres_worker.py - Real-Time PostgreSQL Ingestion & Sync Worker.
Mengambil data broadcast dari PostgreSQL (staging), mengekstrak informasi jam
tangan menggunakan Hybrid NER (spaCy + regex rules), menyimpannya ke RAG DuckDB
secara inkremental (idempotent), dan memicu continuous learning secara otomatis.
Mendukung mode One-Shot Sync dan Daemon Polling kontinu.
"""

import sys
import json
import time
import signal
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.extractor import default_extractor, WatchItem, ExtractionReport
from src.rag_engine import default_rag
from src.trainer import append_to_training_buffer, retrain_from_jsonl
from src.config import (
    DB_HOST,
    DB_PORT,
    DB_DATABASE,
    DB_USERNAME,
    DB_PASSWORD,
    DB_TABLE,
    SYNC_STATE_PATH,
    SYNC_INTERVAL,
    SYNC_BATCH_SIZE,
    AUTO_TRAIN_EVERY,
    AUTO_EXPAND_EVERY,
    TRAIN_ITER,
    UNPARSED_PATH,
    GEMINI_MODEL,
)

# Pewarnaan terminal
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
DIM = "\033[2m"


def save_unparsed_candidates(unparsed_list: List[Dict[str, Any]], out_path: Path = UNPARSED_PATH):
    """Menyimpan format unparsed ke antrean JSONL untuk audit dan ekspansi Gemini."""
    if not unparsed_list:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_records: Dict[str, Dict[str, Any]] = {}

    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        txt = item.get("raw_text", "").strip()
                        if txt:
                            existing_records[txt] = item
        except Exception:
            pass

    for unp in unparsed_list:
        txt = unp.get("raw_text", "").strip()
        if not txt:
            continue
        if txt in existing_records:
            existing_records[txt]["frequency"] = existing_records[txt].get("frequency", 1) + 1
        else:
            unp["frequency"] = 1
            existing_records[txt] = unp

    with open(out_path, "w", encoding="utf-8") as f:
        for item in existing_records.values():
            f.write(json.dumps(item, default=str, ensure_ascii=False) + "\n")


class PostgresSyncWorker:
    def __init__(
        self,
        rag_engine=default_rag,
        extractor=default_extractor,
        state_path: Path = SYNC_STATE_PATH,
    ):
        self.rag = rag_engine
        self.extractor = extractor
        self.state_path = state_path
        self.is_attached = False
        self.running = True
        self.pg_conn = duckdb.connect(":memory:")

    def _ensure_pg_attached(self):
        """Memastikan koneksi postgres terpasang di koneksi DuckDB in-memory terpisah."""
        if self.is_attached:
            return

        try:
            self.pg_conn.execute("SET enable_progress_bar = false;")
        except Exception:
            pass

        try:
            # Periksa apakah pg_db sudah terpasang
            self.pg_conn.execute("LOAD postgres;")
        except Exception:
            # Coba install jika belum terinstall
            self.pg_conn.execute("INSTALL postgres; LOAD postgres;")

        conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_DATABASE} user={DB_USERNAME} password={DB_PASSWORD}"
        attach_query = f"ATTACH '{conn_str}' AS pg_db (TYPE postgres, READ_ONLY);"
        
        try:
            self.pg_conn.execute(attach_query)
            self.is_attached = True
        except Exception as e:
            if "already exists" in str(e).lower() or "database with name pg_db" in str(e).lower():
                self.is_attached = True
            else:
                raise e

    def load_state(self) -> Dict[str, Any]:
        """Membaca watermark sinkronisasi terakhir dari file JSON."""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                return data
            except Exception:
                pass
        return {
            "last_processed_created_at": None,
            "last_processed_id": None,
            "total_synced_messages": 0,
            "total_synced_watches": 0,
            "last_sync_timestamp": None,
        }

    def save_state(self, state: Dict[str, Any]):
        """Menyimpan watermark sinkronisasi secara aman (atomic write)."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state["last_sync_timestamp"] = datetime.now().isoformat()
        temp_file = self.state_path.with_suffix(".tmp")
        temp_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_file.replace(self.state_path)

    def reset_state(self):
        """Mereset watermark sinkronisasi ke posisi awal."""
        if self.state_path.exists():
            self.state_path.unlink()
        print(f"{YELLOW}🔄 Watermark PostgreSQL Sync direset ke awal.{RESET}")

    def fetch_batch(self, last_created_at: Optional[str], last_id: Optional[str], limit: int) -> List[Dict[str, Any]]:
        """Mengambil batch rekaman pesan baru dari PostgreSQL berdasarkan watermark."""
        self._ensure_pg_attached()

        columns = [
            "id", "message_id", "session_id", "chat_id", "chat_jid", "chat_type", "chat_name",
            "sender_jid", "sender_phone", "sender_push_name", "sender_is_verified", "sender_is_business",
            "fulltext_message", "intention", "media_path", "media_url", "thumbnail", "mimetype",
            "timestamp", "created_at", "dealer_alias"
        ]
        col_sql = ", ".join(columns)

        if last_created_at and last_id:
            where_clause = f"""
                WHERE (created_at > TIMESTAMP '{last_created_at}')
                   OR (created_at = TIMESTAMP '{last_created_at}' AND id > '{last_id}')
            """
        elif last_created_at:
            where_clause = f"WHERE created_at > TIMESTAMP '{last_created_at}'"
        else:
            where_clause = ""

        query = f"""
            SELECT {col_sql}
            FROM pg_db.{DB_TABLE}
            {where_clause}
            ORDER BY created_at ASC, id ASC
            LIMIT {limit};
        """

        rows = self.pg_conn.execute(query).fetchall()
        result = []
        for r in rows:
            d = dict(zip(columns, r))
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.strftime("%Y-%m-%d %H:%M:%S.%f")
            result.append(d)
        return result

    def get_remote_stats(self) -> Dict[str, Any]:
        """Mengambil statistik cepat dari tabel PostgreSQL jarak jauh."""
        self._ensure_pg_attached()
        q = f"SELECT count(*), min(created_at), max(created_at) FROM pg_db.{DB_TABLE};"
        total, min_date, max_date = self.pg_conn.execute(q).fetchone()
        return {
            "total_broadcasts": total,
            "min_created_at": str(min_date) if min_date else None,
            "max_created_at": str(max_date) if max_date else None,
        }

    def process_batch(
        self,
        batch: List[Dict[str, Any]],
        state: Dict[str, Any],
        verbose: bool = False,
        auto_train_every: int = AUTO_TRAIN_EVERY,
    ) -> Tuple[int, int, int]:
        """
        Mengekstrak jam dari tiap pesan, mengindeks ke DuckDB, dan memperbarui state.
        Mengembalikan: (total_pesan, total_jam_ditemukan, total_jam_baru_diindeks)
        """
        if not batch:
            return 0, 0, 0

        batch_watches: List[WatchItem] = []
        batch_unparsed = []
        messages_processed = 0

        for row in batch:
            msg_text = row.get("fulltext_message", "")
            if not msg_text:
                continue

            report: ExtractionReport = self.extractor.extract_broadcast_detailed(msg_text, metadata=row)
            messages_processed += 1

            if report.items:
                batch_watches.extend(report.items)
                if verbose:
                    dealer = row.get("dealer_alias") or row.get("sender_push_name") or "Dealer"
                    phone = row.get("sender_phone") or "-"
                    print(f"  {CYAN}• [{dealer} | {phone}]{RESET} Lolos {len(report.items)} jam:")
                    for it in report.items:
                        p_str = f"{it.currency} {it.price_num:,.0f}" if it.price_num else "-"
                        print(f"      - {BOLD}[{it.brand}]{RESET} {it.reference:<14} | Dial: {it.dial or '-':<10} | {p_str}")

            if report.unparsed_lines:
                batch_unparsed.extend(report.unparsed_lines)

            # Update pointer ke row terakhir
            last_created = row.get("created_at")
            if isinstance(last_created, datetime):
                state["last_processed_created_at"] = last_created.strftime("%Y-%m-%d %H:%M:%S.%f")
            elif last_created:
                state["last_processed_created_at"] = str(last_created)
            state["last_processed_id"] = str(row.get("id"))

        # 1. Simpan jam ke DuckDB RAG
        inserted_count = 0
        if batch_watches:
            inserted_count = self.rag.index_items(batch_watches)
            # Masukkan ke buffer data latih spaCy
            append_to_training_buffer(batch_watches)

        # 2. Simpan antrean unparsed
        if batch_unparsed:
            save_unparsed_candidates(batch_unparsed)

        # 3. Update state agregat
        state["total_synced_messages"] = state.get("total_synced_messages", 0) + messages_processed
        state["total_synced_watches"] = state.get("total_synced_watches", 0) + inserted_count
        self.save_state(state)

        return messages_processed, len(batch_watches), inserted_count

    def check_and_trigger_auto_expand(self, auto_expand_every: int = AUTO_EXPAND_EVERY):
        """
        Mengecek antrean unparsed_candidates.jsonl.
        Jika mencapai ambang batas (default 10), picu Gemini Flash untuk analisis pola,
        buat aturan baru, dan jalankan retrain spaCy NER (20 iterasi).
        """
        if auto_expand_every <= 0 or not UNPARSED_PATH.exists():
            return

        try:
            lines = [l for l in UNPARSED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) >= auto_expand_every:
                print(f"\n{MAGENTA}{BOLD}🤖 [AUTONOMOUS EXPANDER] Terkumpul {len(lines)} format unparsed di antrean! (Target pemicu: {auto_expand_every}){RESET}")
                print(f"{CYAN}🧠 Menganalisis {auto_expand_every} format unparsed teratas via Gemini AI ({GEMINI_MODEL})...{RESET}")
                from src.llm_pattern_expander import expand_patterns_with_gemini
                res = expand_patterns_with_gemini(max_lines=auto_expand_every)
                if res.get("status") == "success":
                    rules_count = res.get("new_rules_count", 0)
                    watches_count = res.get("extracted_watches_count", 0)
                    print(f"{GREEN}✔ Berhasil: {watches_count} jam baru diekstrak, {rules_count} aturan regex baru ditambahkan.{RESET}")
                    print(f"{MAGENTA}{BOLD}⚡ [ACTIVE LEARNING] Melatih ulang model spaCy NER ({TRAIN_ITER} iterasi)...{RESET}")
                    # Tutup koneksi DuckDB saat retraining agar proses pembaca/search/script lain leluasa
                    self.rag.close()
                    retrain_from_jsonl(n_iter=TRAIN_ITER)
        except Exception as e:
            print(f"⚠️ [Autonomous Expander] Gagal memproses format unparsed: {e}")

    def run_sync(
        self,
        limit: Optional[int] = None,
        batch_size: int = SYNC_BATCH_SIZE,
        verbose: bool = False,
        auto_train_every: int = AUTO_TRAIN_EVERY,
        auto_expand_every: int = AUTO_EXPAND_EVERY,
    ):
        """Sinkronisasi One-Shot: menarik pesan baru hingga limit atau hingga habis."""
        self._ensure_pg_attached()
        state = self.load_state()

        remote_stats = self.get_remote_stats()
        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║         📡 POSTGRESQL STAGING -> RAG ENGINE SYNCHRONIZATION WORKER               ║{RESET}")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")
        print(f"🌐 Remote Host       : {DB_HOST}:{DB_PORT} / {DB_DATABASE}")
        print(f"📋 Remote Table      : {DB_TABLE} ({remote_stats['total_broadcasts']:,} total broadcast)")
        print(f"📍 Watermark Awal    : Created At: {state.get('last_processed_created_at') or 'Mulai dari awal'}")
        if state.get("last_processed_id"):
            print(f"                       ID Terakhir: {state.get('last_processed_id')}")
        print(f"📦 Total di DuckDB   : {self.rag.count_watches():,} jam")
        print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────────{RESET}\n")

        total_msgs = 0
        total_watches_found = 0
        total_inserted = 0
        accumulated_watches_for_train = 0

        while self.running:
            current_batch_limit = batch_size
            if limit is not None:
                remaining = limit - total_msgs
                if remaining <= 0:
                    break
                current_batch_limit = min(batch_size, remaining)

            last_created = state.get("last_processed_created_at")
            last_id = state.get("last_processed_id")

            batch = self.fetch_batch(last_created, last_id, limit=current_batch_limit)
            if not batch:
                print(f"{GREEN}✔ Seluruh pesan di remote database sudah tersinkronisasi sempurna!{RESET}")
                break

            m_count, w_found, w_ins = self.process_batch(
                batch,
                state,
                verbose=verbose,
                auto_train_every=auto_train_every,
            )

            total_msgs += m_count
            total_watches_found += w_found
            total_inserted += w_ins
            accumulated_watches_for_train += w_ins

            total_db = self.rag.count_watches()
            print(f"📥 [{total_msgs:,} pesan diproses] +{w_found} jam terekstrak (+{w_ins} baru di DuckDB) | Total DB: {total_db:,} jam")

            # Cek ambang batas auto-retrain
            if auto_train_every > 0 and accumulated_watches_for_train >= auto_train_every:
                print(f"\n{MAGENTA}{BOLD}⚡ [ACTIVE LEARNING] Ambang batas {auto_train_every} jam baru tercapai! Melatih spaCy...{RESET}")
                retrain_from_jsonl(n_iter=TRAIN_ITER)
                accumulated_watches_for_train = 0

            # Cek ambang batas antrean format unparsed untuk Gemini AI Auto-Expansion
            self.check_and_trigger_auto_expand(auto_expand_every)

            if len(batch) < current_batch_limit:
                # Sudah mencapai ujung data PostgreSQL
                break

        print(f"\n{GREEN}{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}")
        print(f"{GREEN}{BOLD}✨ RINGKASAN SINKRONISASI POSTGRESQL SELESAI{RESET}")
        print(f"  • Pesan Ditambahkan       : {BOLD}{total_msgs:,} pesan{RESET}")
        print(f"  • Jam Tangan Terekstrak   : {BOLD}{total_watches_found:,} unit{RESET}")
        print(f"  • Jam Baru Tersimpan DuckDB: {BOLD}{total_inserted:,} unit{RESET}")
        print(f"  • Total Database Saat Ini : {BOLD}{self.rag.count_watches():,} unit{RESET}")
        print(f"  • Watermark Sekarang      : {state.get('last_processed_created_at')}")
        print(f"{GREEN}{BOLD}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")

    def run_polling_daemon(
        self,
        interval: float = SYNC_INTERVAL,
        batch_size: int = SYNC_BATCH_SIZE,
        verbose: bool = False,
        auto_train_every: int = AUTO_TRAIN_EVERY,
        auto_expand_every: int = AUTO_EXPAND_EVERY,
    ):
        """Mode Daemon: Polling terus menerus setiap `interval` detik secara live."""
        def handle_sigint(signum, frame):
            print(f"\n{YELLOW}🛑 Menerima sinyal berhenti (SIGINT/Ctrl+C). Menyelesaikan pekerjaan...{RESET}")
            self.running = False

        signal.signal(signal.SIGINT, handle_sigint)

        self._ensure_pg_attached()
        state = self.load_state()

        print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
        print(f"{CYAN}{BOLD}║         🔄 POSTGRESQL REAL-TIME CONTINUOUS POLLING DAEMON STARTED        ║{RESET}")
        print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")
        print(f"🌐 Remote Host     : {DB_HOST}:{DB_PORT} / {DB_DATABASE}")
        print(f"⏱️ Interval Polling : Setiap {interval} detik")
        print(f"📦 Batch Size       : {batch_size} pesan per tarikan")
        print(f"📍 Watermark Awal  : {state.get('last_processed_created_at') or 'Mulai dari awal'}")
        print(f"Tekan {BOLD}Ctrl+C{RESET} kapan saja untuk keluar dengan aman.\n")

        accumulated_watches_for_train = 0

        while self.running:
            last_created = state.get("last_processed_created_at")
            last_id = state.get("last_processed_id")

            batch = self.fetch_batch(last_created, last_id, limit=batch_size)

            if batch:
                m_count, w_found, w_ins = self.process_batch(
                    batch,
                    state,
                    verbose=verbose,
                    auto_train_every=auto_train_every,
                )
                accumulated_watches_for_train += w_ins
                now_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{now_str}] 📥 +{m_count} pesan baru | +{w_found} jam terekstrak (+{w_ins} DuckDB) | Total DB: {self.rag.count_watches():,}")

                if auto_train_every > 0 and accumulated_watches_for_train >= auto_train_every:
                    print(f"\n{MAGENTA}{BOLD}⚡ [ACTIVE LEARNING] Auto-retraining spaCy ({accumulated_watches_for_train} sample)...{RESET}")
                    self.rag.close()
                    retrain_from_jsonl(n_iter=TRAIN_ITER)
                    accumulated_watches_for_train = 0

                # Cek ambang batas antrean format unparsed untuk Gemini AI Auto-Expansion
                self.check_and_trigger_auto_expand(auto_expand_every)

                # Jika batch penuh, langsung tarik sisa berikutnya tanpa menunggu interval
                if len(batch) >= batch_size:
                    continue

            # Lepas file lock database saat idle agar proses pembaca/search/script lain leluasa mengakses DuckDB
            self.rag.close()

            # Menunggu interval berikutnya
            now_str = datetime.now().strftime("%H:%M:%S")
            sys.stdout.write(f"\r{DIM}[{now_str}] 🟢 Up-to-date. Menunggu pesan masuk baru di PostgreSQL ({interval}s)...{RESET} ")
            sys.stdout.flush()

            # Tidur dengan resolusi kecil agar SIGINT responsif
            sleep_step = 0.5
            elapsed = 0.0
            while self.running and elapsed < interval:
                time.sleep(sleep_step)
                elapsed += sleep_step

        self.rag.close()
        print(f"\n{GREEN}✔ Daemon berhenti dengan bersih. Watermark tersimpan di {self.state_path}. Sampai jumpa!{RESET}\n")


default_sync_worker = PostgresSyncWorker()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="PostgreSQL Real-Time Watch Broadcast Sync Worker")
    parser.add_argument("--limit", type=int, default=None, help="Batas maksimal pesan yang disinkronkan")
    parser.add_argument("--batch-size", type=int, default=SYNC_BATCH_SIZE, help="Ukuran batch SQL")
    parser.add_argument("--poll", action="store_true", help="Jalankan dalam mode daemon polling kontinu")
    parser.add_argument("--interval", type=float, default=SYNC_INTERVAL, help="Interval polling dalam detik")
    parser.add_argument("--reset", action="store_true", help="Reset watermark sinkronisasi ke awal")
    parser.add_argument("--auto-train-every", type=int, default=AUTO_TRAIN_EVERY, help=f"Ambang batas sample baru untuk auto-retrain spaCy (default: {AUTO_TRAIN_EVERY})")
    parser.add_argument("--auto-expand-every", type=int, default=AUTO_EXPAND_EVERY, help=f"Ambang batas antrean unparsed untuk panggil Gemini & retrain (default: {AUTO_EXPAND_EVERY})")
    parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan detail tiap jam yang terekstrak")

    args = parser.parse_args()

    if args.reset:
        default_sync_worker.reset_state()

    if args.poll:
        default_sync_worker.run_polling_daemon(
            interval=args.interval,
            batch_size=args.batch_size,
            verbose=args.verbose,
            auto_train_every=args.auto_train_every,
            auto_expand_every=args.auto_expand_every,
        )
    else:
        default_sync_worker.run_sync(
            limit=args.limit,
            batch_size=args.batch_size,
            verbose=args.verbose,
            auto_train_every=args.auto_train_every,
            auto_expand_every=args.auto_expand_every,
        )


if __name__ == "__main__":
    main()

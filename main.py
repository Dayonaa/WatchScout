"""
main.py - Unified Entrypoint & CLI Hub for Watch Dealer NER & RAG Engine.
Menyediakan antarmuka terpusat untuk Search, Stream Simulator, Training, dan Pattern Review.
"""

import sys
import argparse
from pathlib import Path

# Pastikan path modul terbaca
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag_engine import default_rag
from src.config import (
    MODEL_DIR,
    UNPARSED_PATH,
    BROADCASTS_CSV_PATH,
    DEVICE_INFO,
    TRAIN_ITER,
    AUTO_TRAIN_EVERY,
    AUTO_EXPAND_EVERY,
    STREAM_SPEED,
    EXPAND_AUTO_RETRAIN,
    EXPAND_MAX,
)


def print_status():
    import json
    print("=" * 70)
    print(" ⌚ WATCH DEALER NER & FAST RAG ENGINE - SYSTEM STATUS")
    print("=" * 70)
    
    hw_text = "🟢 NVIDIA GPU (CUDA) Aktif" if DEVICE_INFO["active_device"] == "cuda" else "💻 CPU Mode (Otomatis beralih ke GPU jika CUDA terdeteksi)"
    print(f"⚡ Hardware Acceleration        : {hw_text}")
    print(f"📦 Total Jam Tersimpan di DuckDB : {default_rag.count_watches():,} unit")
    
    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"🧠 Versi Base Model spaCy Active : v{m.get('version', 'Unknown')}")
            latest = m.get("latest_training", {})
            print(f"   • Weight Fingerprint          : {latest.get('weight_fingerprint', '-')}")
            print(f"   • Final Loss                  : {latest.get('final_loss', '-')}")
            print(f"   • Total Data Training         : {latest.get('dataset_examples', '-')} contoh")
        except Exception:
            pass

    unparsed_path = UNPARSED_PATH
    if unparsed_path.exists():
        lines = [l for l in unparsed_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"🔬 Format Belum Dikenal (Queue)   : {len(lines)} variasi unik di antrean")

    from src.config import DB_HOST, DB_DATABASE, DB_TABLE, SYNC_STATE_PATH
    if SYNC_STATE_PATH.exists():
        try:
            st = json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))
            last_dt = st.get("last_processed_created_at") or "Awal"
            print(f"📡 Status Sinkronisasi PostgreSQL: Aktif ({DB_HOST}/{DB_DATABASE})")
            print(f"   • Watermark Terakhir          : {last_dt}")
            print(f"   • Total Sinkronisasi          : {st.get('total_synced_messages', 0):,} pesan diproses")
        except Exception:
            pass
    else:
        print(f"📡 Remote PostgreSQL Staging    : Siap disinkronkan ({DB_HOST}/{DB_DATABASE} -> {DB_TABLE})")

    print("=" * 70)
    print("\nPerintah Cepat yang Tersedia:")
    print("  • python main.py sync --limit 50              : Sinkronisasi 50 pesan dari PostgreSQL")
    print("  • python main.py sync --poll --interval 5     : Jalankan live polling background daemon")
    print("  • python main.py search \"Rolex 126300 green\"   : Cari jam di database RAG")
    print("  • python main.py stream --limit 30            : Jalankan simulasi stream WhatsApp (CSV)")
    print("  • python main.py review                       : Tinjau format belum dikenal")
    print("  • python main.py expand --retrain             : Panggil Gemini untuk ekspansi regex & auto-retrain")
    print("  • python main.py train                        : Latih ulang model spaCy NER")
    print("  • python main.py interactive                  : Masuk ke shell pencarian interaktif\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for Luxury Watch Dealer Extraction & RAG Search",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Perintah yang ingin dijalankan")

    # 1. Search Subcommand
    search_parser = subparsers.add_parser("search", help="Pencarian semantik jam tangan di RAG DuckDB")
    search_parser.add_argument("query", type=str, nargs="?", default=None, help="Kata kunci pencarian (misal: '126300 green')")
    search_parser.add_argument("--max-price", type=float, default=None, help="Batas harga maksimal (HKD)")
    search_parser.add_argument("--min-year", type=int, default=None, help="Tahun minimal")
    search_parser.add_argument("--dial", type=str, default=None, help="Filter warna dial")
    search_parser.add_argument("--limit", type=int, default=5, help="Jumlah hasil pencarian (default: 5)")

    # 2. Stream Subcommand
    stream_parser = subparsers.add_parser("stream", help="Simulasi aliran chat broadcast dealer WhatsApp")
    stream_parser.add_argument("--speed", type=float, default=STREAM_SPEED, help=f"Delay antar pesan dalam detik (default: {STREAM_SPEED})")
    stream_parser.add_argument("--limit", type=int, default=30, help="Jumlah pesan yang ingin disimulasikan (default: 30)")
    stream_parser.add_argument("--auto-train-every", type=int, default=AUTO_TRAIN_EVERY, help=f"Ambang batas item baru untuk auto-retrain (default: {AUTO_TRAIN_EVERY})")
    stream_parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan detail semua baris jam")

    # 3. Review Subcommand
    review_parser = subparsers.add_parser("review", help="Audit format belum dikenal untuk perancangan regex baru")
    review_parser.add_argument("--clear", action="store_true", help="Kosongkan seluruh antrean format unparsed")

    # 4. Train Subcommand
    train_parser = subparsers.add_parser("train", help="Melatih ulang model spaCy NER dari buffer data")
    train_parser.add_argument("--iter", type=int, default=TRAIN_ITER, help=f"Jumlah iterasi pelatihan (default: {TRAIN_ITER})")

    # 5. Expand Subcommand (Gemini AI Self-Expansion)
    expand_parser = subparsers.add_parser("expand", help="Panggil Gemini untuk analisis format unparsed & buat aturan baru")
    expand_parser.add_argument("--max", type=int, default=EXPAND_MAX, help="Jumlah baris unparsed yang dianalisis (default: semua baris di antrean)")
    expand_parser.add_argument("--retrain", action=argparse.BooleanOptionalAction, default=EXPAND_AUTO_RETRAIN, help=f"Otomatis retrain spaCy setelah aturan dibuat (default: {EXPAND_AUTO_RETRAIN})")

    # 6. Interactive Subcommand
    subparsers.add_parser("interactive", help="Buka shell interaktif pencarian jam")

    # 7. SQL Subcommand (Direct SQL Inspection)
    sql_parser = subparsers.add_parser("sql", help="Jalankan query SQL langsung ke database DuckDB")
    sql_parser.add_argument("query", type=str, nargs="?", default="SELECT brand, count(*) as total FROM watches GROUP BY brand ORDER BY total DESC", help="Query SQL (default: agregasi per brand)")

    # 8. Sync Subcommand (PostgreSQL Ingestion & Polling Daemon)
    sync_parser = subparsers.add_parser("sync", help="Tarik & sinkronkan data broadcast langsung dari PostgreSQL staging")
    sync_parser.add_argument("--limit", type=int, default=None, help="Batas maksimal pesan yang disinkronkan (default: semua pesan pending)")
    sync_parser.add_argument("--batch-size", type=int, default=100, help="Jumlah baris per batch tarikan SQL (default: 100)")
    sync_parser.add_argument("--poll", action="store_true", help="Jalankan continuous polling loop secara live (daemon)")
    sync_parser.add_argument("--interval", type=float, default=5.0, help="Interval polling dalam detik saat mode --poll (default: 5.0)")
    sync_parser.add_argument("--reset", action="store_true", help="Reset watermark sinkronisasi ke pesan terawal")
    sync_parser.add_argument("--auto-train-every", type=int, default=AUTO_TRAIN_EVERY, help=f"Ambang batas sample baru untuk retrain spaCy (default: {AUTO_TRAIN_EVERY})")
    sync_parser.add_argument("--auto-expand-every", type=int, default=AUTO_EXPAND_EVERY, help=f"Ambang batas antrean unparsed untuk panggil Gemini & retrain (default: {AUTO_EXPAND_EVERY})")
    sync_parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan detail tiap jam yang terekstrak")

    args = parser.parse_args()

    if args.command == "sync":
        from src.postgres_worker import default_sync_worker
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

    elif args.command == "search":
        import app
        if args.query:
            app.run_search(args.query, limit=args.limit, max_price=args.max_price, min_year=args.min_year, dial=args.dial)
        else:
            app.interactive_shell()

    elif args.command == "stream":
        import stream_simulator
        csv_file = BROADCASTS_CSV_PATH
        stream_simulator.run_simulation(
            csv_path=csv_file,
            speed=args.speed,
            limit=args.limit,
            auto_train_every=args.auto_train_every,
            verbose=args.verbose
        )

    elif args.command == "review":
        if getattr(args, "clear", False):
            from src.config import UNPARSED_PATH
            UNPARSED_PATH.write_text("", encoding="utf-8")
            print(f"\n🧹 Antrean unparsed ({UNPARSED_PATH}) berhasil dikosongkan!\n")
        else:
            import stream_simulator
            stream_simulator.review_unparsed_candidates()

    elif args.command == "expand":
        from src.llm_pattern_expander import expand_patterns_with_gemini
        from src.trainer import retrain_from_jsonl
        res = expand_patterns_with_gemini(max_lines=args.max)
        if res.get("status") == "success" and args.retrain:
            print("\n🔄 Memulai auto-retrain spaCy di CPU dengan data baru...")
            retrain_from_jsonl(n_iter=TRAIN_ITER)

    elif args.command == "train":
        from src.trainer import retrain_from_jsonl
        retrain_from_jsonl(n_iter=args.iter)

    elif args.command == "interactive":
        import app
        app.interactive_shell()

    elif args.command == "sql":
        try:
            print(f"\n📊 Query: {args.query}\n")
            default_rag.conn.sql(args.query).show()
            print()
        except Exception as e:
            print(f"❌ Error SQL: {e}")

    else:
        print_status()


if __name__ == "__main__":
    main()
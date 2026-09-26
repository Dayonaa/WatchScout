#!/usr/bin/env python3
"""
scripts/read_db.py - Utility Script untuk Membaca & Memeriksa Database WatchScout
Mendukung pembacaan database lokal DuckDB (hasil ekstraksi jam) dan remote PostgreSQL (pesan broadcast mentah).
Aman dijalankan bersamaan dengan background daemon (non-blocking read_only mode).
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DB_PATH,
    SYNC_STATE_PATH,
    MODEL_DIR,
    DB_HOST,
    DB_PORT,
    DB_DATABASE,
    DB_USERNAME,
    DB_PASSWORD,
    DB_TABLE,
)


def connect_duckdb_safely(retries: int = 5, delay: float = 0.5):
    """Membuka koneksi DuckDB dalam mode read-only dengan retry otomatis."""
    import duckdb
    for attempt in range(retries):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except Exception as e:
            if ("Conflicting lock" in str(e) or "Could not set lock" in str(e)) and attempt < retries - 1:
                time.sleep(delay)
                continue
            raise e


def show_overview():
    """Menampilkan ringkasan status database, model, dan sinkronisasi."""
    print("=" * 80)
    print("📊 WATCHSCOUT DATABASE & SYSTEM OVERVIEW")
    print("=" * 80)

    # 1. DuckDB Info
    if not DB_PATH.exists():
        print(f"❌ File database belum ada di: {DB_PATH}")
        return

    conn = connect_duckdb_safely()
    try:
        total_watches = conn.execute("SELECT COUNT(*) FROM watches").fetchone()[0]
        brands = conn.execute("""
            SELECT brand, COUNT(*) as cnt 
            FROM watches 
            WHERE brand IS NOT NULL 
            GROUP BY brand 
            ORDER BY cnt DESC 
            LIMIT 6
        """).fetchall()
        
        prices = conn.execute("""
            SELECT 
                currency, 
                MIN(price_num) as min_p, 
                AVG(price_num) as avg_p, 
                MAX(price_num) as max_p, 
                COUNT(*) as cnt
            FROM watches 
            WHERE price_num IS NOT NULL AND price_num > 0
            GROUP BY currency
        """).fetchall()

        dealers_cnt = conn.execute("SELECT COUNT(DISTINCT sender_phone) FROM watches WHERE sender_phone IS NOT NULL").fetchone()[0]

    finally:
        conn.close()

    print(f"📁 Lokasi DuckDB     : {DB_PATH} ({DB_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"⌚ Total Jam Ekstrak : {total_watches:,} unit")
    print(f"📱 Total Dealer Aktif: {dealers_cnt} nomor dealer")

    if brands:
        print("\n🏆 Top Brand Terbanyak:")
        for b, count in brands:
            print(f"   • {b:<20}: {count:,} unit")

    if prices:
        print("\n💰 Rentang Harga:")
        for cur, pmin, pavg, pmax, cnt in prices:
            print(f"   • {cur} ({cnt:,} unit): Min {pmin:,.0f} | Rata-rata {pavg:,.0f} | Max {pmax:,.0f}")

    # 2. Sync Watermark Info
    import json
    if SYNC_STATE_PATH.exists():
        try:
            state = json.loads(SYNC_STATE_PATH.read_text(encoding="utf-8"))
            print(f"\n📍 Watermark PostgreSQL:")
            print(f"   • Waktu Terakhir : {state.get('last_processed_created_at', '-')}")
            print(f"   • ID Terakhir    : {state.get('last_processed_id', '-')}")
            print(f"   • Terakhir Sync  : {state.get('last_synced_at', '-')}")
        except Exception:
            pass

    # 3. Model Info
    meta_path = MODEL_DIR / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"\n🧠 Model spaCy NER:")
            print(f"   • Versi Aktif    : v{meta.get('version', '0.1.0')}")
            print(f"   • Waktu Pelatihan: {meta.get('last_trained', '-')}")
            print(f"   • Total Training : {meta.get('total_train_runs', 0)} kali")
        except Exception:
            pass

    print("=" * 80 + "\n")


def list_recent_watches(limit: int = 15, brand_filter: str = None, ref_filter: str = None):
    """Menampilkan daftar jam tangan terbaru dari DuckDB."""
    conn = connect_duckdb_safely()
    try:
        query = """
            SELECT 
                brand, series, reference, dial, year, condition, currency, price_num, dealer_alias, sender_phone, broadcast_time
            FROM watches
            WHERE 1=1
        """
        params = []
        if brand_filter:
            query += " AND LOWER(brand) LIKE ?"
            params.append(f"%{brand_filter.lower()}%")
        if ref_filter:
            query += " AND LOWER(reference) LIKE ?"
            params.append(f"%{ref_filter.lower()}%")

        query += " ORDER BY broadcast_time DESC NULLS LAST, id DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        print("❌ Tidak ada data jam yang cocok dengan kriteria pencarian.")
        return

    print(f"\n⌚ DAFTAR {len(rows)} JAM TANGAN TERBARU:")
    print("-" * 125)
    header = f"{'BRAND':<14} | {'SERIES':<18} | {'REF':<14} | {'DIAL':<12} | {'YEAR':<5} | {'KONDISI':<7} | {'HARGA':<16} | {'DEALER / PHONE'}"
    print(header)
    print("-" * 125)

    for r in rows:
        b, s, ref, dial, yr, cond, cur, price, d_alias, s_phone, b_time = r
        b_str = (b or "-")[:13]
        s_str = (s or "-")[:17]
        ref_str = (ref or "-")[:13]
        dial_str = (dial or "-")[:11]
        yr_str = str(yr) if yr else "-"
        cond_str = cond or "-"
        price_str = f"{cur} {price:,.0f}" if price else "N/A"
        dealer_str = f"{d_alias or ''} ({s_phone or '-'})".strip()

        print(f"{b_str:<14} | {s_str:<18} | {ref_str:<14} | {dial_str:<12} | {yr_str:<5} | {cond_str:<7} | {price_str:<16} | {dealer_str}")

    print("-" * 125 + "\n")


def execute_custom_sql(sql: str):
    """Menjalankan query SQL kustom pada database DuckDB."""
    conn = connect_duckdb_safely()
    try:
        res = conn.execute(sql)
        cols = [desc[0] for desc in res.description]
        rows = res.fetchall()
        print(f"\n🔍 HASIL SQL QUERY ({len(rows)} baris):")
        print(" | ".join(cols))
        print("-" * 100)
        for r in rows[:50]:
            print(" | ".join(str(val) for val in r))
        if len(rows) > 50:
            print(f"... dan {len(rows) - 50} baris lainnya.")
        print("-" * 100 + "\n")
    finally:
        conn.close()


def inspect_remote_postgres(limit: int = 5):
    """Membaca pesan broadcast mentah langsung dari PostgreSQL remote."""
    import duckdb
    print(f"\n🌐 Menghubungkan ke PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_DATABASE}...")
    mem_conn = duckdb.connect(":memory:")
    try:
        mem_conn.execute("INSTALL postgres; LOAD postgres;")
        conn_str = f"host={DB_HOST} port={DB_PORT} dbname={DB_DATABASE} user={DB_USERNAME} password={DB_PASSWORD}"
        mem_conn.execute(f"ATTACH '{conn_str}' AS pg (TYPE POSTGRES, READ_ONLY);")

        total = mem_conn.execute(f"SELECT COUNT(*) FROM pg.{DB_TABLE}").fetchone()[0]
        print(f"📋 Total Broadcast di PostgreSQL: {total:,} pesan")

        rows = mem_conn.execute(f"""
            SELECT id, created_at, sender_phone, left(raw_message, 80) as msg_preview
            FROM pg.{DB_TABLE}
            ORDER BY created_at DESC
            LIMIT {limit}
        """).fetchall()

        print(f"\n📩 {len(rows)} Pesan Terakhir di PostgreSQL:")
        for r in rows:
            print(f"  [{r[1]}] ID: {r[0]} | Phone: {r[2]}")
            print(f"     Preview: {repr(r[3])}\n")

    except Exception as e:
        print(f"❌ Gagal membaca PostgreSQL: {e}")
    finally:
        mem_conn.close()


def main():
    parser = argparse.ArgumentParser(description="WatchScout Database Reader & Inspector")
    parser.add_argument("--overview", "-o", action="store_true", help="Tampilkan ringkasan statistik DB & sistem")
    parser.add_argument("--list", "-l", type=int, nargs="?", const=15, default=None, help="Tampilkan N jam tangan terbaru (default: 15)")
    parser.add_argument("--brand", "-b", type=str, default=None, help="Filter berdasarkan brand jam (cth: Rolex, Omega)")
    parser.add_argument("--ref", "-r", type=str, default=None, help="Filter berdasarkan nomor referensi (cth: 116500, 5711)")
    parser.add_argument("--sql", "-s", type=str, default=None, help="Jalankan query SQL bebas ke DuckDB")
    parser.add_argument("--pg", action="store_true", help="Periksa data mentah di database remote PostgreSQL")

    args = parser.parse_args()

    if args.pg:
        inspect_remote_postgres(limit=5)
        return

    if args.sql:
        execute_custom_sql(args.sql)
        return

    if args.list is not None or args.brand or args.ref:
        limit = args.list if args.list is not None else 15
        list_recent_watches(limit=limit, brand_filter=args.brand, ref_filter=args.ref)
        return

    # Default action: tampilkan overview dan 10 jam terbaru
    show_overview()
    list_recent_watches(limit=10)


if __name__ == "__main__":
    main()

"""
app.py - Antarmuka Utama Base Model Watch NER & RAG Search Engine.
Mendukung parsing broadcast massal, indexing otomatis, dan pencarian instan via CLI.
"""

import sys
import time
import argparse
from pathlib import Path

from src.extractor import default_extractor
from src.rag_engine import default_rag


def parse_and_index_file(file_path: str):
    """Mengekstrak file broadcast dan menyimpannya ke database RAG DuckDB."""
    p = Path(file_path)
    if not p.exists():
        print(f"❌ File tidak ditemukan: {file_path}")
        return

    print(f"\n📖 Membaca file: {file_path}")
    raw_text = p.read_text(encoding="utf-8")

    t0 = time.time()
    count = default_rag.index_broadcast(raw_text, default_extractor)
    elapsed = time.time() - t0

    print(f"✅ Berhasil mengekstrak & mengindeks {count} jam tangan dalam {elapsed:.2f} detik!")
    print(f"📊 Total jam tersimpan di DuckDB: {default_rag.count_watches():,}")


def run_search(query: str, limit: int = 5, max_price: float = None, min_year: int = None, dial: str = None):
    """Menjalankan query RAG hybrid search."""
    print("=" * 80)
    print(f"🔍 PENCARIAN RAG: \"{query}\"")
    if max_price:
        print(f"  • Filter Maksimal Harga: HKD {max_price:,.0f}")
    if min_year:
        print(f"  • Filter Minimal Tahun : {min_year}")
    if dial:
        print(f"  • Filter Warna Dial    : {dial}")
    print("=" * 80)

    t0 = time.time()
    results = default_rag.search(
        query=query,
        limit=limit,
        max_price=max_price,
        min_year=min_year,
        dial=dial
    )
    search_time = (time.time() - t0) * 1000

    if not results:
        print("Tidak ada jam yang cocok dengan kriteria pencarian.")
        return

    print(f"\nMenemukan {len(results)} hasil ({search_time:.2f} ms):\n")
    print(f"{'SCORE':<7} | {'REFERENCE':<18} | {'BRAND / SERIES':<26} | {'DIAL':<12} | {'HARGA':<14} | {'KONDISI':<7} | {'DEALER / WHATSAPP'}")
    print("-" * 115)
    for r in results:
        price_str = f"{r['currency']} {r['price']:,.0f}" if r['price'] else "-"
        dial_str = (r['dial'] or "-")[:12]
        brand_series = f"{r['brand']} {r['series'] or ''}".strip()[:26]
        dealer_info = f"{r['dealer_alias'] or r['sender_name'] or 'Dealer'} ({r['sender_phone'] or 'No Phone'})"
        media_str = ""
        if r.get("media_path") or r.get("media_url"):
            m = r.get("media_path") or r.get("media_url")
            media_str = f"\n        └─ 📸 Media: {m}"
        print(f"{r['score']:<7.4f} | {r['reference']:<18} | {brand_series:<26} | {dial_str:<12} | {price_str:<14} | {r['condition']:<7} | {dealer_info}{media_str}")
    print("-" * 115)


def interactive_shell():
    """Mode interaktif terminal untuk pencarian langsung."""
    print("=" * 70)
    print(" ⌚ WATCH DEALER NER & RAG SEARCH ENGINE")
    print("=" * 70)
    print(f"Total Database Saat Ini: {default_rag.count_watches()} jam tangan.")
    print("Ketik kata kunci pencarian Anda (atau 'exit' untuk keluar).")
    print("Contoh:")
    print("  • Overseas blue dial")
    print("  • green dial rose gold")
    print("  • 4500V 2024")
    print("=" * 70 + "\n")

    while True:
        try:
            q = input("🔎 Cari Jam > ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Sampai jumpa!")
                break

            run_search(q, limit=5)
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nSelesai.")
            break


def main():
    parser = argparse.ArgumentParser(description="Watch Broadcast NER & Fast RAG Engine")
    parser.add_argument("--parse", type=str, default=None, help="Parse file broadcast dan indeks ke database")
    parser.add_argument("--search", type=str, default=None, help="Pencarian query RAG cepat")
    parser.add_argument("--max-price", type=float, default=None, help="Filter harga maksimal (HKD)")
    parser.add_argument("--min-year", type=int, default=None, help="Filter tahun minimal")
    parser.add_argument("--dial", type=str, default=None, help="Filter warna dial (Blue, Green, Silver, Black, dll.)")
    parser.add_argument("--limit", type=int, default=5, help="Jumlah hasil pencarian (default: 5)")
    parser.add_argument("-i", "--interactive", action="store_true", help="Buka shell interaktif")

    args = parser.parse_args()

    if args.parse:
        parse_and_index_file(args.parse)
        return

    if args.search:
        run_search(
            query=args.search,
            limit=args.limit,
            max_price=args.max_price,
            min_year=args.min_year,
            dial=args.dial
        )
        return

    # Default: interactive shell
    interactive_shell()


if __name__ == "__main__":
    main()

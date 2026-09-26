"""
stream_simulator.py - Real-Time Dealer Broadcast Stream Simulator & Continuous Active Learner.
Mensimulasikan chat WhatsApp dealer masuk secara live, mengekstrak entitas jam
secara instan di CPU, mengindeks ke DuckDB Vector RAG dengan metadata kontak,
dan memicu auto-retrain model spaCy saat ambang batas data baru tercapai.
Dilengkapi fitur penampung 'Format Belum Dikenal' untuk penemuan pola & pembuatan regex baru.
"""

import os
import csv
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Pastikan path modul terbaca
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.extractor import default_extractor, WatchItem, ExtractionReport
from src.rag_engine import default_rag
from src.trainer import append_to_training_buffer, retrain_from_jsonl
from src.config import (
    UNPARSED_PATH as UNPARSED_LOG_PATH,
    BROADCASTS_CSV_PATH,
    AUTO_EXPAND_EVERY,
    TRAIN_ITER,
    GEMINI_MODEL,
)

# Kode ANSI untuk pewarnaan terminal yang elegan
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"
DIM = "\033[2m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║        ⚡ LUXURY WATCH DEALER BROADCAST REAL-TIME STREAM SIMULATOR ⚡            ║
║           CPU Ultra-Fast Ingestion • Hybrid RAG • Continuous Learning            ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}
"""
    print(banner)


def format_price(currency: str, price_num: Any) -> str:
    if not price_num:
        return f"{DIM}-{RESET}"
    return f"{GREEN}{BOLD}{currency} {price_num:,.0f}{RESET}"


def save_unparsed_candidates(unparsed_list: List[Dict[str, Any]], out_path: Path = UNPARSED_LOG_PATH):
    """Menyimpan atau memperbarui antrean baris chat unparsed ke format JSONL."""
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
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def analyze_unparsed_hint(text: str) -> str:
    """Memberikan petunjuk analisis cepat mengapa baris ini belum terurai."""
    import re
    t_lower = text.lower()
    if re.search(r"(\$|hkd|hld|k\b|[0-9]{5,})", t_lower):
        if re.search(r"\b(iwc|omega|cartier|rolex|patek|hublot|breitling|panerai)\b", t_lower):
            return "💡 Ada harga & nama brand, namun nomor model/ref belum cocok dengan regex."
        return "💡 Terdeteksi harga/angka, kemungkinan nomor referensi belum terdaftar di kamus."
    if any(w in t_lower for w in ["box", "paper", "card", "set", "link", "dial"]):
        return "💡 Kemungkinan parts / kelengkapan jam (box, papers, link tambahan)."
    return "💡 Pola teks non-standar atau chat percakapan dealer."


def review_unparsed_candidates(log_path: Path = UNPARSED_LOG_PATH, top_n: int = 15):
    """Menampilkan ringkasan audit format belum dikenal untuk perancangan regex baru."""
    print(f"\n{CYAN}{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}{BOLD}║         🔬 AUDIT FORMAT BELUM DIKENAL (BAHAN PERANCANGAN REGEX BARU)             ║{RESET}")
    print(f"{CYAN}{BOLD}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")

    if not log_path.exists():
        print(f"{YELLOW}Belum ada data unparsed di {log_path.resolve()}. Jalankan simulasi stream terlebih dahulu.{RESET}\n")
        return

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"{GREEN}Tidak ada baris unparsed! Semua format terbaca dengan sempurna.{RESET}\n")
        return

    # Urutkan berdasarkan frekuensi kemunculan terbanyak
    records.sort(key=lambda x: x.get("frequency", 1), reverse=True)

    print(f"Total Pola Unparsed Terkumpul: {BOLD}{len(records)} variasi unik{RESET}")
    print(f"Menampilkan Top {min(top_n, len(records))} pola paling sering muncul di broadcast dealer:\n")

    for idx, rec in enumerate(records[:top_n], start=1):
        txt = rec.get("raw_text", "")
        freq = rec.get("frequency", 1)
        dealer = rec.get("dealer_alias") or rec.get("sender_phone") or "Unknown"
        hint = analyze_unparsed_hint(txt)

        print(f"{BOLD}[{idx:02d}] Muncul {freq}x | Dari: {YELLOW}{dealer}{RESET}")
        print(f"     Teks Mentah: {CYAN}\"{txt}\"{RESET}")
        print(f"     {hint}")
        print()

    print(f"{DIM}Berkas sumber: {log_path.resolve()}{RESET}\n")


def run_simulation(
    csv_path: Path,
    speed: float = 0.2,
    limit: int = 30,
    auto_train_every: int = 25,
    skip_empty: bool = True,
    verbose: bool = False
):
    print_banner()
    print(f"📂 Dataset Sumber       : {BOLD}{csv_path.resolve()}{RESET}")
    print(f"⏱️ Kecepatan Delay      : {BOLD}{speed} detik / chat{RESET}")
    print(f"🎯 Batas Pesan Stream   : {BOLD}{limit} pesan{RESET}")
    print(f"🔄 Auto-Retrain Threshold: {BOLD}Setiap {auto_train_every} jam terekstrak{RESET}")
    print(f"📦 Total Database Awal  : {BOLD}{default_rag.count_watches()} item{RESET}\n")

    if not csv_path.exists():
        print(f"{RED}Error: File {csv_path} tidak ditemukan!{RESET}")
        return

    msg_count = 0
    total_extracted = 0
    total_lines_analyzed = 0
    total_skipped = 0
    buffer_counter = 0
    global_reasons: Dict[str, int] = {}
    all_unparsed_candidates: List[Dict[str, Any]] = []

    with open(csv_path, mode="r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            msg_count += 1
            if limit and msg_count > limit:
                break

            msg_text = row.get("fulltext_message", "").strip()
            if not msg_text:
                continue

            # Ekstraksi entitas dari pesan dealer dengan laporan kelolosan detail
            t0 = time.perf_counter()
            report: ExtractionReport = default_extractor.extract_row_detailed(row)
            items = report.items
            latency_ms = (time.perf_counter() - t0) * 1000

            total_lines_analyzed += report.total_lines
            total_skipped += report.skipped_count
            for reason, cnt in report.skipped_reasons.items():
                global_reasons[reason] = global_reasons.get(reason, 0) + cnt

            # Kumpulkan kandidat format unparsed
            if report.unparsed_lines:
                all_unparsed_candidates.extend(report.unparsed_lines)

            if not items and skip_empty:
                continue

            # Metadata pengirim
            dealer = row.get("dealer_alias") or row.get("sender_push_name") or "Unknown Dealer"
            phone = row.get("sender_phone") or "No Phone"
            chat = row.get("chat_name") or "Direct/Group"

            # Header pesan (kompak)
            print(f"{CYAN}──────────────────────────────────────────────────────────────────────────────────{RESET}")
            print(f"{BOLD}[💬 Chat #{msg_count:03d}/{limit}]{RESET} {YELLOW}{BOLD}{dealer}{RESET} ({phone})")
            print(f"   {DIM}Grup: {chat} | Latensi: {latency_ms:.2f} ms | Total Baris: {report.total_lines}{RESET}")

            if items:
                print(f"   {GREEN}✔ {BOLD}{len(items)} Jam Lolos{RESET} ({CYAN}{report.skipped_count} baris dilewati/filter{RESET})")

                # Mode ringkas vs verbose
                if verbose:
                    for it in items:
                        cond_color = GREEN if it.condition == "NEW" else YELLOW
                        dial_str = f"{CYAN}{it.dial}{RESET}" if it.dial else f"{DIM}-{RESET}"
                        print(f"     • {BOLD}[{it.brand}]{RESET} {it.reference:<16} | Dial: {dial_str:<16} | {cond_color}{it.condition:<4}{RESET} | {format_price(it.currency, it.price_num)}")
                else:
                    sample_items = items[:2]
                    sample_strs = []
                    for it in sample_items:
                        p_str = f"{it.currency} {it.price_num:,.0f}" if it.price_num else "-"
                        sample_strs.append(f"{BOLD}[{it.brand}]{RESET} {it.reference} ({p_str})")

                    remaining = len(items) - len(sample_items)
                    summary_line = " • ".join(sample_strs)
                    if remaining > 0:
                        summary_line += f" ... {DIM}(+{remaining} jam lainnya){RESET}"
                    print(f"   📋 Contoh: {summary_line}")

                # 1. Simpan ke RAG Vector Database DuckDB
                inserted = default_rag.index_items(items)
                total_in_db = default_rag.count_watches()
                print(f"   💾 {BLUE}DuckDB Indexed: +{inserted} item | Total Database: {total_in_db:,} jam{RESET}")

                # 2. Masukkan ke buffer Active Learning
                added_to_buffer = append_to_training_buffer(items)
                buffer_counter += added_to_buffer
                total_extracted += len(items)

                # 3. Pemicu Auto-Retraining jika threshold tercapai
                if buffer_counter >= auto_train_every:
                    print(f"\n{MAGENTA}{BOLD}⚡ [ACTIVE LEARNING TRIGGERED] Ambang batas {auto_train_every} sample baru tercapai!{RESET}")
                    retrain_from_jsonl(n_iter=10)
                    buffer_counter = 0

            else:
                print(f"   {DIM}(Dilewati: tidak ditemukan jam valid pada pesan ini){RESET}")

            # Delay simulasi
            if speed > 0:
                time.sleep(speed)

    # Simpan akumulasi format belum dikenal ke berkas audit
    if all_unparsed_candidates:
        save_unparsed_candidates(all_unparsed_candidates)
        try:
            if UNPARSED_LOG_PATH.exists():
                unp_lines = [l for l in UNPARSED_LOG_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
                if AUTO_EXPAND_EVERY > 0 and len(unp_lines) >= AUTO_EXPAND_EVERY:
                    print(f"\n{MAGENTA}{BOLD}🤖 [AUTONOMOUS EXPANDER] Terkumpul {len(unp_lines)} format unparsed di antrean! (Ambang batas: {AUTO_EXPAND_EVERY}){RESET}")
                    print(f"{CYAN}🧠 Menganalisis {AUTO_EXPAND_EVERY} format unparsed via Gemini AI ({GEMINI_MODEL})...{RESET}")
                    from src.llm_pattern_expander import expand_patterns_with_gemini
                    res = expand_patterns_with_gemini(max_lines=AUTO_EXPAND_EVERY)
                    if res.get("status") == "success":
                        print(f"\n{MAGENTA}{BOLD}⚡ [ACTIVE LEARNING] Auto-retraining model spaCy ({TRAIN_ITER} iterasi)...{RESET}")
                        retrain_from_jsonl(n_iter=TRAIN_ITER)
        except Exception as e:
            print(f"⚠️ [Autonomous Expander] Gagal memproses: {e}")

    # Laporan Audit Komprehensif
    pass_pct = (total_extracted / total_lines_analyzed * 100) if total_lines_analyzed > 0 else 0
    skip_pct = (total_skipped / total_lines_analyzed * 100) if total_lines_analyzed > 0 else 0

    print(f"\n{CYAN}╔══════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║               📊 LAPORAN AUDIT EKSTRAKSI & KELOLOSAN DATA STREAM                ║{RESET}")
    print(f"{CYAN}╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"  • Total Chat WhatsApp Masuk : {BOLD}{msg_count} pesan{RESET}")
    print(f"  • Total Baris Teks Dianalisis: {BOLD}{total_lines_analyzed:,} baris{RESET}")
    print(f"  • {GREEN}{BOLD}Jam LOLOS Masuk Database  : {total_extracted:,} unit ({pass_pct:.1f}%){RESET}")
    print(f"  • {YELLOW}{BOLD}Baris DILEWATI (Terfilter) : {total_skipped:,} baris ({skip_pct:.1f}%){RESET}")
    print(f"\n  {BOLD}Rincian Baris yang Dilewati:{RESET}")
    for reason, count in sorted(global_reasons.items(), key=lambda x: x[1], reverse=True):
        label = reason.replace("_", " ").title()
        print(f"    - {label:<25}: {count:,} baris")

    print(f"\n  • Total Jam Aktif di DuckDB RAG: {BOLD}{GREEN}{default_rag.count_watches():,} jam{RESET}")
    print(f"  • 📁 Antrean Regex Baru        : {BOLD}{UNPARSED_LOG_PATH.resolve()}{RESET}")
    print(f"    {DIM}(Ketik 'python stream_simulator.py --review-unparsed' untuk melihat analisis pola baru){RESET}")
    print(f"{CYAN}══════════════════════════════════════════════════════════════════════════════════{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Simulasi Stream Chat Dealer Jam Real-Time")
    parser.add_argument("--csv", type=str, default=str(BROADCASTS_CSV_PATH), help="Path ke CSV broadcast dealer")
    parser.add_argument("--speed", type=float, default=0.2, help="Delay antar pesan dalam detik (default: 0.2)")
    parser.add_argument("--limit", type=int, default=30, help="Jumlah pesan yang ingin disimulasikan (default: 30)")
    parser.add_argument("--auto-train-every", type=int, default=25, help="Ambang batas item baru untuk auto-retrain model (default: 25)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Tampilkan semua baris jam tanpa dipotong")
    parser.add_argument("--keep-empty", action="store_true", help="Jangan abaikan pesan yang kosong dari jam")
    parser.add_argument("--review-unparsed", action="store_true", help="Tampilkan analisis format belum dikenal untuk perancangan regex baru")

    args = parser.parse_args()

    if args.review_unparsed:
        review_unparsed_candidates()
        return

    csv_file = Path(args.csv)

    run_simulation(
        csv_path=csv_file,
        speed=args.speed,
        limit=args.limit,
        auto_train_every=args.auto_train_every,
        skip_empty=not args.keep_empty,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()

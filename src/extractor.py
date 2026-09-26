"""
src/extractor.py - High-Performance Watch Broadcast Information Extractor.
Didesain untuk inferensi ultra-cepat di CPU tanpa GPU.
Mendukung multi-brand, stateful condition tracking, inline override (n7/n8/n9),
penanganan glued tokens, dan pelacakan metadata kontak dealer WhatsApp.
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reference_decoder import (
    decode_reference,
    extract_explicit_dial,
    parse_price,
    split_glued_token,
)


class WatchItem(BaseModel):
    brand: str = "Rolex"
    series: Optional[str] = None
    reference: str
    dial: Optional[str] = None
    material: Optional[str] = None
    year: Optional[int] = None
    condition: str = "USED"
    currency: str = "HKD"
    price_raw: Optional[str] = None
    price_num: Optional[float] = None
    raw_text: str
    message_id: Optional[str] = None
    sender_phone: Optional[str] = None
    sender_name: Optional[str] = None
    dealer_alias: Optional[str] = None
    chat_name: Optional[str] = None
    broadcast_time: Optional[str] = None


# Pola regex komposit untuk nomor referensi jam tangan berbagai merek
REF_PATTERNS = [
    # 1. Cartier (WJPN..., WGTA..., WSSA..., etc.)
    re.compile(r"\b(W[A-Z]{3}[0-9A-Z]{4,6})\b", re.IGNORECASE),
    # 2. Tudor (2836c1a0-0105, 79030N, etc.)
    re.compile(r"\b(2836[0-9A-Z\-]+|2860[0-9A-Z\-]+|79[0-9]{3}[A-Z\-]*|25600[A-Z\-]*)\b", re.IGNORECASE),
    # 3. Vacheron & Patek dengan Slash (4500v/110a-b483, 82035/000R-9359, 7300/1200A, 5711/1A)
    re.compile(r"\b([0-9]{4,5}[A-Z]?(?:[/|-][0-9A-Z]{3,4}[A-Z]?)(?:[- ][0-9A-Z]{4})?)\b", re.IGNORECASE),
    # 4. Patek & AP Format Pendek (5330G, 15500ST, 26331ST)
    re.compile(r"\b(5[0-9]{3}[A-Z]|15[45][0-9]{2}[A-Z]{2}|26[234][0-9]{2}[A-Z]{2}|16202[A-Z]{2})\b", re.IGNORECASE),
    # 5. Rolex Standar 5-6 Digit + Bezel/Gold Suffix (126300, 116500LN, 126610LV, 336934, 126067, 128238A, 52508-0006)
    re.compile(r"\b([1235][0-9]{4,5}(?:-[0-9]{4})?(?:LN|LV|BLRO|BLNR|CHNR|GRNR|LB|JC|[AG])?)\b", re.IGNORECASE),
]

# Regex pola tahun (1990 - 2035) atau format kartu 2025/6
YEAR_PATTERN = re.compile(r"\b(199\d|20[0-3]\d)(?:/[0-9]{1,2})?\b")

# Regex pola harga dealer WhatsApp
# Prioritas:
# 1. Dengan simbol mata uang di depan: $85,300, $437000, HKD 200k, hld219000
# 2. Dengan mata uang/multiplier di belakang: 445000HKD, 182k, 265k
# 3. Standalone angka harga tinggi (>= 10,000) untuk menghindari tahun (1990-2035) dan card code (N8, N9)
PRICE_PREFIX_PATTERN = re.compile(
    r"(?:[\$]|HKD|HLD|HKHKD|USD|SGD|RMB|CNY)\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)\s*([KkMm])?",
    re.IGNORECASE
)
PRICE_SUFFIX_PATTERN = re.compile(
    r"([0-9]{1,3}(?:,[0-9]{3})+|[0-9]+(?:\.[0-9]+)?)\s*(?:[KkMm]|HKD|HLD|HKHKD|USD|SGD)\b",
    re.IGNORECASE
)
PRICE_STANDALONE_PATTERN = re.compile(
    r"\b([1-9][0-9]{0,2}(?:,[0-9]{3})+|[1-9][0-9]{4,7})\b"  # 10,000 s/d 99,999,999 atau dengan koma
)

# Regex inline condition & dealer card slang
INLINE_NEW_PATTERN = re.compile(r"\b(NEW|BNIB|UNWORN|LIKE\s*NEW|N[789]|NEW\s*CARD)\b", re.IGNORECASE)
INLINE_USED_PATTERN = re.compile(r"\b(USED|PRE-OWNED|PREOWNED|2ND|9[5-9]%)\b", re.IGNORECASE)

# Header condition lines
HEADER_USED_PATTERN = re.compile(r"^(used(?:\s*/\s*used)*|pre-?owned|\*+used\*+)$", re.IGNORECASE)
HEADER_NEW_PATTERN = re.compile(r"^(new(?:\s*/\s*new)*|brand\s*new|bnib|\*+new\*+)$", re.IGNORECASE)

# Filter baris non-jam tangan: aksesoris, box, dan permintaan beli (looking to buy/WTB)
ACCESSORY_PATTERN = re.compile(
    r"\b(winder|travel\s*case|travel\s*roll|presentation\s*box|watch\s*box|booklet|card\s*holder|wallet|cufflinks)\b",
    re.IGNORECASE
)
INQUIRY_PATTERN = re.compile(
    r"\b(looking\s*to\s*buy|looking\s*for|want\s*to\s*buy|wtb|sold\s*order|need\s*new|price\s*updated|hong\s*kong\s*in\s*stock)\b",
    re.IGNORECASE
)

# Header brand lines
HEADER_BRAND_MAP = {
    "ROLEX": "Rolex",
    "CARTIER": "Cartier",
    "TUDOR": "Tudor",
    "VACHERON": "Vacheron Constantin",
    "VC": "Vacheron Constantin",
    "AUDEMARS": "Audemars Piguet",
    "AP": "Audemars Piguet",
    "PATEK": "Patek Philippe",
    "PP": "Patek Philippe",
    "OMEGA": "Omega",
}


class ExtractionReport(BaseModel):
    items: List[WatchItem] = []
    total_lines: int = 0
    passed_count: int = 0
    skipped_count: int = 0
    skipped_reasons: Dict[str, int] = Field(default_factory=dict)
    unparsed_lines: List[Dict[str, Any]] = Field(default_factory=list)


class WatchBroadcastExtractor:
    def __init__(self, default_brand: str = "Rolex"):
        self.default_brand = default_brand

    def extract_line(
        self,
        line: str,
        current_condition: str = "USED",
        current_brand: Optional[str] = None
    ) -> Optional[WatchItem]:
        """
        Mengekstrak 1 baris broadcast dealer menjadi objek WatchItem terstruktur.
        """
        line_clean = line.strip()
        if not line_clean:
            return None

        # Abaikan baris divider / emoji dekorasi saja
        if re.match(r"^[_=\-*\s🔥🇭🇰🪵✨]+$", line_clean):
            return None

        # Preprocessing: Cek apakah ada glued token (misal: 126300ombre green atau 336934blue)
        tokens = line_clean.split()
        glued_dial_hint = None
        processed_tokens = []
        for tok in tokens:
            cleaned_tok, dial_hint = split_glued_token(tok)
            if dial_hint:
                glued_dial_hint = dial_hint
                processed_tokens.extend([cleaned_tok, dial_hint])
            else:
                processed_tokens.append(cleaned_tok)

        processed_line = " ".join(processed_tokens)

        # 1. Ekstraksi Nomor Referensi
        raw_ref = None
        ref_span = None

        for pat in REF_PATTERNS:
            match = pat.search(processed_line)
            if match:
                raw_ref = match.group(1).strip()
                ref_span = match.span()
                break

        if not raw_ref:
            return None

        # Abaikan referensi yang murni hanya tahun (misal 2024 atau 2025)
        if re.match(r"^(199\d|20[0-3]\d)$", raw_ref):
            return None

        # Decode referensi dengan domain knowledge
        decoded_info = decode_reference(raw_ref)

        # Tentukan brand: jika decoded_info memiliki brand spesifik, utamakan decoded_info
        resolved_brand = decoded_info.get("brand") or current_brand or self.default_brand

        # Teks setelah referensi untuk ekstraksi entitas lain
        after_ref_text = processed_line[ref_span[1]:]

        # 2. Ekstraksi Tahun
        year_match = YEAR_PATTERN.search(after_ref_text)
        year_val = None
        if year_match:
            try:
                year_val = int(year_match.group(1))
            except (ValueError, TypeError):
                pass

        # 3. Penentuan Kondisi: 2-Tier Stateful Resolution
        # Prioritas 1: Inline Override (NEW, BNIB, N7/N8/N9 vs USED)
        if INLINE_NEW_PATTERN.search(after_ref_text) or INLINE_NEW_PATTERN.search(processed_line):
            cond_val = "NEW"
        elif INLINE_USED_PATTERN.search(after_ref_text) or INLINE_USED_PATTERN.search(processed_line):
            cond_val = "USED"
        else:
            # Prioritas 2: Header Inheritance
            cond_val = current_condition

        # 4. Ekstraksi Harga
        price_raw = None
        currency_val = "HKD"
        price_num_val = None

        search_price_text = after_ref_text if after_ref_text.strip() else processed_line
        best_price_match = None

        # Prioritas 1: Simbol di depan ($437000, HKD200k, hld219000)
        p_pre = PRICE_PREFIX_PATTERN.search(search_price_text)
        if p_pre:
            best_price_match = p_pre.group(0).strip()
        else:
            # Prioritas 2: Simbol/Multiplier di belakang (445000HKD, 182k, 265k)
            p_suf = PRICE_SUFFIX_PATTERN.search(search_price_text)
            if p_suf:
                best_price_match = p_suf.group(0).strip()
            else:
                # Prioritas 3: Standalone angka harga tinggi (>= 10,000)
                # Pastikan bukan tahun dan bukan referensi
                for p_num in PRICE_STANDALONE_PATTERN.finditer(search_price_text):
                    num_cand = p_num.group(1).strip()
                    if raw_ref and raw_ref in num_cand:
                        continue
                    if year_val and str(year_val) == num_cand:
                        continue
                    best_price_match = num_cand
                    break

        if best_price_match:
            price_raw = best_price_match
            currency_val, price_num_val = parse_price(price_raw)

        # 5. Ekstraksi Dial
        # Urutan prioritas:
        # a. Explicit dial dalam after_ref_text (Ombre Green, Blue, Wimbledon, etc.)
        # b. Glued dial hint yang terdeteksi saat pemisahan token
        # c. Decoded dial dari suffix referensi
        explicit_dial = extract_explicit_dial(after_ref_text)
        final_dial = explicit_dial or glued_dial_hint or decoded_info.get("dial")

        return WatchItem(
            brand=resolved_brand,
            series=decoded_info.get("series"),
            reference=decoded_info.get("reference", raw_ref.upper()),
            dial=final_dial,
            material=decoded_info.get("material"),
            year=year_val,
            condition=cond_val,
            currency=currency_val,
            price_raw=price_raw,
            price_num=price_num_val,
            raw_text=line_clean,
        )

    def extract_broadcast_detailed(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExtractionReport:
        """
        Memproses teks broadcast dan mengembalikan ExtractionReport lengkap:
        - items yang LOLOS (valid watch item)
        - statistik total baris, jumlah lolos vs dilewati
        - rincian alasan kenapa suatu baris dilewati (Header, Aksesoris, WTB, dsb.)
        """
        report = ExtractionReport()
        current_condition = "USED"
        current_brand = self.default_brand

        lines = text.splitlines()
        report.total_lines = len(lines)

        for raw_line in lines:
            line = raw_line.strip()
            if not line or re.match(r"^[_=\-*\s🔥🇭🇰🪵✨]+$", line):
                report.skipped_count += 1
                report.skipped_reasons["divider_atau_kosong"] = report.skipped_reasons.get("divider_atau_kosong", 0) + 1
                continue

            # 1. Cek apakah baris ini adalah AKSESORIS / BOX
            if ACCESSORY_PATTERN.search(line):
                report.skipped_count += 1
                report.skipped_reasons["aksesoris_atau_box"] = report.skipped_reasons.get("aksesoris_atau_box", 0) + 1
                continue

            # 2. Cek apakah baris ini adalah CHATTER / PERMINTAAN BELI (WTB)
            if INQUIRY_PATTERN.search(line):
                report.skipped_count += 1
                report.skipped_reasons["permintaan_beli_wtb"] = report.skipped_reasons.get("permintaan_beli_wtb", 0) + 1
                continue

            # 3. Cek apakah baris ini adalah HEADER KONDISI
            clean_hdr = line.lower().replace("/", " ").strip()
            if HEADER_USED_PATTERN.match(line) or clean_hdr in ["used", "used used", "used used used"]:
                current_condition = "USED"
                report.skipped_count += 1
                report.skipped_reasons["header_kondisi"] = report.skipped_reasons.get("header_kondisi", 0) + 1
                continue
            if HEADER_NEW_PATTERN.match(line) or clean_hdr in ["new", "brand new", "bnib"]:
                current_condition = "NEW"
                report.skipped_count += 1
                report.skipped_reasons["header_kondisi"] = report.skipped_reasons.get("header_kondisi", 0) + 1
                continue

            # 4. Cek apakah baris ini adalah HEADER BRAND
            upper_line = line.upper().strip()
            is_brand_hdr = False
            for brand_key, brand_val in HEADER_BRAND_MAP.items():
                if upper_line == brand_key or upper_line.startswith(brand_key + " "):
                    current_brand = brand_val
                    is_brand_hdr = True
                    break
            if is_brand_hdr:
                report.skipped_count += 1
                report.skipped_reasons["header_merek"] = report.skipped_reasons.get("header_merek", 0) + 1
                continue

            # 5. Parse item baris jam
            item = self.extract_line(
                line,
                current_condition=current_condition,
                current_brand=current_brand
            )

            if item:
                # Lampirkan metadata pengirim jika tersedia
                if metadata:
                    item.message_id = metadata.get("message_id")
                    item.sender_phone = metadata.get("sender_phone")
                    item.sender_name = metadata.get("sender_push_name")
                    item.dealer_alias = metadata.get("dealer_alias")
                    item.chat_name = metadata.get("chat_name")
                    item.broadcast_time = metadata.get("timestamp") or metadata.get("created_at")

                report.items.append(item)
                report.passed_count += 1
            else:
                report.skipped_count += 1
                report.skipped_reasons["format_tidak_dikenal"] = report.skipped_reasons.get("format_tidak_dikenal", 0) + 1
                # Simpan baris yang punya potensi data jam (mengandung huruf/angka) ke antrean audit
                if len(line) >= 4 and any(c.isalnum() for c in line):
                    report.unparsed_lines.append({
                        "raw_text": line,
                        "sender_phone": metadata.get("sender_phone") if metadata else None,
                        "dealer_alias": metadata.get("dealer_alias") if metadata else None,
                        "chat_name": metadata.get("chat_name") if metadata else None,
                        "timestamp": (metadata.get("timestamp") or metadata.get("created_at")) if metadata else None,
                    })

        return report

    def extract_broadcast(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[WatchItem]:
        """
        Wrapper kompatibilitas: memproses teks broadcast dan mengembalikan List[WatchItem] yang lolos.
        """
        report = self.extract_broadcast_detailed(text, metadata=metadata)
        return report.items

    def extract_row_detailed(self, row: Dict[str, Any]) -> ExtractionReport:
        """
        Mengekstrak 1 baris CSV dengan laporan kelolosan lengkap.
        """
        msg_text = row.get("fulltext_message", "")
        if not msg_text:
            return ExtractionReport()
        return self.extract_broadcast_detailed(msg_text, metadata=row)

    def extract_row(self, row: Dict[str, Any]) -> List[WatchItem]:
        """
        Mengekstrak pesan dari 1 baris CSV watch_broadcasts.
        """
        return self.extract_row_detailed(row).items


default_extractor = WatchBroadcastExtractor()

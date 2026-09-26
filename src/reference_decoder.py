"""
src/reference_decoder.py - Modul pengetahuan domain jam tangan mewah.
Mendekodekan dial, material, series, brand, dan normalisasi harga untuk multi-merek
(Rolex, Vacheron Constantin, Cartier, Tudor, Patek Philippe, Audemars Piguet).
"""

import re
from typing import Optional, Dict, Any, Tuple


# ==========================================
# 1. KAMUS DIAL & WARNA
# ==========================================

# Suffix Referensi Vacheron Constantin
VC_DIAL_MAP = {
    # Overseas Blue Dials
    "B128": "Blue",
    "B546": "Blue",
    "B148": "Blue",
    "B590": "Blue",
    "B984": "Blue",
    "B170": "Blue",
    "B442": "Blue",
    "B576": "Blue",
    "B969": "Blue",
    "B971": "Blue",
    "H101": "Petroleum Blue",

    # Overseas Silver / White Dials
    "B483": "Silver",
    "B333": "Silver",
    "B591": "Silver",
    "B978": "Silver",
    "B968": "Silver",
    "B336": "Silver",
    "B334": "Silver",
    "B344": "Silver",
    "B085": "Silver",
    "B622": "Silver",
    "B077": "Silver",
    "B078": "Silver",
    "B052": "Silver",
    "B051": "Silver",
    "B141": "Silver",
    "B497": "Silver",
    "B109": "Silver",
    "B289": "Silver",
    "B441": "Silver",
    "B487": "Silver",
    "H014": "Silver",
    "H134": "Silver",

    # Overseas & Traditionnelle Green Dials
    "B952": "Green",
    "B966": "Green",
    "B967": "Green",
    "B979": "Green",
    "B980": "Green",
    "B965": "Green",

    # Black Dials
    "B127": "Black",
    "9338": "Black",

    # Brown / Chocolate Dials
    "B074": "Brown",
    "B705": "Brown",

    # Pink / Salmon Dials
    "B592": "Salmon",
    "H015": "Salmon",
}

# Regex pola dial eksplisit & nama gaul dealer (Wimbledon, Ombre, Tiffany, etc.)
EXPLICIT_DIAL_PATTERNS = [
    # Slang & Spesifik
    (r"\b(ombre\s*green|omber\s*green)\b", "Ombre Green"),
    (r"\b(wimbledon)\b", "Wimbledon"),
    (r"\b(tiffany|turquoise)\b", "Tiffany / Turquoise"),
    (r"\b(ice\s*blue)\b", "Ice Blue"),
    (r"\b(mint\s*green)\b", "Mint Green"),
    (r"\b(olive\s*green|olive)\b", "Olive Green"),
    (r"\b(meteorite|mete)\b", "Meteorite"),
    (r"\b(starbucks)\b", "Green (Starbucks)"),
    (r"\b(pepsi)\b", "Pepsi Bezel"),
    (r"\b(batman|batgirl)\b", "Blue/Black (Batman)"),
    (r"\b(root\s*beer)\b", "Root Beer"),
    (r"\b(kermit|hulk)\b", "Green (Hulk/Kermit)"),
    (r"\b(multi|celebration)\b", "Multi / Celebration"),
    (r"\b(sunburst\s*blue)\b", "Sunburst Blue"),
    (r"\b(rhodium|slate|grey|gray)\b", "Slate / Rhodium"),
    (r"\b(champagne|gold\s*dial)\b", "Champagne"),
    (r"\b(chocolate|brown)\b", "Chocolate / Brown"),
    (r"\b(salmon|pink)\b", "Salmon"),
    (r"\b(skeleton|openworked)\b", "Skeleton"),
    (r"\b(diamond\s*dial|baguette|diamond)\b", "Diamond Dial"),

    # Standar Warna
    (r"\b(blue|biru)\b", "Blue"),
    (r"\b(black|hitam|blk)\b", "Black"),
    (r"\b(green|hijau)\b", "Green"),
    (r"\b(silver|perak)\b", "Silver"),
    (r"\b(white|putih|wht)\b", "White"),
]


# ==========================================
# 2. KAMUS MATERIAL
# ==========================================

VC_MATERIAL_MAP = {
    "A": "Stainless Steel",
    "R": "Rose Gold",
    "G": "White Gold",
    "J": "Yellow Gold",
    "P": "Platinum",
    "T": "Titanium",
    "M": "Two-Tone (Steel/Gold)",
    "C": "Ceramic",
}

# Angka terakhir pada nomor referensi Rolex menunjukkan material case
ROLEX_MATERIAL_DIGIT = {
    "0": "Stainless Steel (Oystersteel)",
    "1": "Everose Rolesor (Steel & Rose Gold)",
    "2": "Rolesium (Steel & Platinum)",
    "3": "Yellow Rolesor (Steel & Yellow Gold)",
    "4": "White Rolesor (Steel & White Gold)",
    "5": "Everose Gold (18k Rose Gold)",
    "6": "Platinum",
    "7": "Titanium (RLX Titanium)",
    "8": "18k Yellow Gold",
    "9": "18k White Gold",
}


# ==========================================
# 3. KAMUS SERI MODEL
# ==========================================

VC_SERIES_PREFIX_MAP = {
    "4500V": "Overseas Self-Winding",
    "4520V": "Overseas Self-Winding",
    "5500V": "Overseas Chronograph",
    "5520V": "Overseas Chronograph",
    "7900V": "Overseas Dual Time",
    "7920V": "Overseas Dual Time",
    "2305V": "Overseas Small Model",
    "1205V": "Overseas Quartz",
    "1225V": "Overseas Quartz",
    "6000V": "Overseas Tourbillon",
    "4600V": "Fiftysix Self-Winding",
    "4605V": "Fiftysix Complete Calendar",
    "4600E": "Fiftysix Self-Winding",
    "4400E": "Fiftysix Day-Date",
    "82035": "Historiques American 1921",
    "4200H": "Historiques 222",
    "81590": "Patrimony Contemporaine",
    "85290": "Patrimony Traditionnelle",
    "81180": "Patrimony Grande Taille",
    "4010T": "Patrimony Retrograde Day-Date",
    "1110U": "Patrimony Manual-Winding",
    "4000E": "Fiftysix Complete Calendar",
    "4000U": "Patrimony Self-Winding",
    "4010U": "Traditionnelle Complete Calendar",
    "82172": "Traditionnelle Manual-Winding",
    "82572": "Traditionnelle Small Seconds",
    "43175": "Traditionnelle Tourbillon",
    "47192": "Patrimony Traditionnelle Chrono",
    "47292": "Patrimony Traditionnelle Perpetual",
    "83020": "Historiques Toledo 1951",
    "83570": "Historiques Ultra-Fine 1955",
    "86060": "Traditionnelle World Time",
    "86050": "Traditionnelle Calibre 2755",
    "89000": "Traditionnelle Tourbillon 14 Days",
    "30130": "Malte Tourbillon",
    "7000M": "Harmony Dual Time",
    "7810S": "Harmony Chronograph",
    "8005F": "Harmony Complete Calendar",
    "47112": "Malte Chronograph",
    "47400": "Malte Dual Time",
    "47450": "Overseas Dual Time",
    "49020": "Malte Tourbillon Regulator",
    "49150": "Overseas Chronograph",
    "25553": "Malte Tonneau",
    "7805S": "Harmony Chronograph",
    "46245": "Les Historiques",
}

ROLEX_SERIES_PREFIX_MAP = {
    # Datejust
    "1263": "Datejust 41",
    "1163": "Datejust II",
    "1262": "Datejust 36",
    "1162": "Datejust 36",
    "2791": "Lady-Datejust 28",
    "2782": "Datejust 31",

    # Daytona
    "1265": "Cosmograph Daytona",
    "1165": "Cosmograph Daytona",

    # Submariner
    "1266": "Submariner Date",
    "1166": "Submariner Date",
    "1240": "Submariner (No Date)",
    "1140": "Submariner (No Date)",

    # GMT-Master II
    "1267": "GMT-Master II",
    "1167": "GMT-Master II",

    # Day-Date
    "2282": "Day-Date 40",
    "2182": "Day-Date II",
    "1282": "Day-Date 36",
    "1182": "Day-Date 36",

    # Sky-Dweller
    "3369": "Sky-Dweller",
    "3269": "Sky-Dweller",
    "3362": "Sky-Dweller",

    # Oyster Perpetual
    "1260": "Oyster Perpetual",
    "1243": "Oyster Perpetual 41",
    "1242": "Oyster Perpetual 34",
    "2772": "Oyster Perpetual 31",

    # Sea-Dweller & Deepsea
    "1366": "Deepsea",
    "12660": "Sea-Dweller",
    "12606": "Deepsea Challenge",

    # Yacht-Master
    "12662": "Yacht-Master 40",
    "22665": "Yacht-Master 42",
    "22667": "Yacht-Master 42",
    "2266": "Yacht-Master 42",
    "26862": "Yacht-Master 37",

    # 1908
    "5250": "Perpetual 1908",
}

CARTIER_PREFIX_MAP = {
    "WJPN": ("Cartier", "Panthère de Cartier", "Rose/Yellow Gold & Diamonds"),
    "WGTA": ("Cartier", "Tank Louis Cartier", "Gold"),
    "WSSA": ("Cartier", "Santos de Cartier", "Stainless Steel"),
    "WSTA": ("Cartier", "Tank Must", "Stainless Steel"),
    "WSBB": ("Cartier", "Ballon Bleu de Cartier", "Stainless Steel"),
    "WJBB": ("Cartier", "Ballon Bleu de Cartier", "Gold & Diamonds"),
    "WGBB": ("Cartier", "Ballon Bleu de Cartier", "Gold"),
    "WGPA": ("Cartier", "Pasha de Cartier", "Gold"),
    "WSPA": ("Cartier", "Pasha de Cartier", "Stainless Steel"),
}

TUDOR_PREFIX_MAP = {
    "2836": ("Tudor", "Tudor Royal", "Stainless Steel"),
    "2860": ("Tudor", "Tudor Royal", "Steel & Gold"),
    "7903": ("Tudor", "Black Bay 58", "Stainless Steel"),
    "7901": ("Tudor", "Black Bay 58", "Gold / Silver"),
    "7923": ("Tudor", "Black Bay", "Stainless Steel"),
    "7936": ("Tudor", "Black Bay Chrono", "Stainless Steel"),
    "2560": ("Tudor", "Pelagos", "Titanium"),
}

PATEK_PREFIX_MAP = {
    "5711": ("Patek Philippe", "Nautilus", "Stainless Steel"),
    "5712": ("Patek Philippe", "Nautilus Moonphase", "Stainless Steel"),
    "5726": ("Patek Philippe", "Nautilus Annual Calendar", "Stainless Steel"),
    "5811": ("Patek Philippe", "Nautilus", "White Gold"),
    "5167": ("Patek Philippe", "Aquanaut", "Stainless Steel"),
    "5168": ("Patek Philippe", "Aquanaut Jumbo", "White Gold"),
    "5968": ("Patek Philippe", "Aquanaut Chronograph", "Stainless Steel"),
    "5330": ("Patek Philippe", "Calatrava World Time", "White Gold"),
    "7300": ("Patek Philippe", "Twenty~4 Automatic", "Stainless Steel"),
    "5205": ("Patek Philippe", "Annual Calendar", "Gold"),
}

AUDEMARS_PREFIX_MAP = {
    "15500": ("Audemars Piguet", "Royal Oak Selfwinding", "Stainless Steel"),
    "15510": ("Audemars Piguet", "Royal Oak 50th Anniversary", "Stainless Steel"),
    "15400": ("Audemars Piguet", "Royal Oak Selfwinding", "Stainless Steel"),
    "16202": ("Audemars Piguet", "Royal Oak 'Jumbo' Extra-Thin", "Stainless Steel"),
    "26331": ("Audemars Piguet", "Royal Oak Chronograph", "Stainless Steel"),
    "26240": ("Audemars Piguet", "Royal Oak Chronograph 50th", "Stainless Steel"),
    "26399": ("Audemars Piguet", "Code 11.59 Minute Repeater", "Gold"),
    "15210": ("Audemars Piguet", "Code 11.59", "Gold / Ceramic"),
}


# ==========================================
# 4. GLUED TOKEN SEPARATOR
# ==========================================

CURRENCY_WORDS = {"HKD", "HLD", "HKHKD", "USD", "SGD", "EUR", "RMB", "CNY", "IDR", "GBP"}

# Pola untuk memisahkan referensi Rolex/jam lain yang menempel ke kata warna dial
# Contoh: 126300ombre green -> 126300, ombre green
# Contoh: 336934blue -> 336934, blue
GLUED_REF_REGEX = re.compile(
    r"^([0-9]{4,6}(?:LN|LV|BLRO|BLNR|CHNR|GRNR|LB|JC)?)([a-zA-Z\s]+)$",
    re.IGNORECASE
)


def split_glued_token(token: str) -> Tuple[str, Optional[str]]:
    """
    Jika sebuah token berisi gabungan referensi dan dial (misal: '126300ombre' atau '336934blue'),
    pisahkan menjadi nomor referensi dan petunjuk dial.
    Jangan memisahkan harga seperti '445000HKD' atau '82500hkd' sebagai dial.
    """
    token_clean = token.strip()
    match = GLUED_REF_REGEX.match(token_clean)
    if match:
        ref_part = match.group(1).strip()
        dial_part = match.group(2).strip()

        # Jika bagian huruf adalah mata uang (HKD, HLD), ini adalah harga, BUKAN dial!
        if dial_part.upper() in CURRENCY_WORDS or dial_part.upper().startswith("HK"):
            return token_clean, None

        # Periksa apakah dial_part adalah kata warna yang valid
        dial_found = extract_explicit_dial(dial_part)
        if dial_found:
            return ref_part, dial_found
        elif len(dial_part) >= 3 and not re.match(r"^[KkMm]$", dial_part):
            # Hanya kembalikan jika bukan huruf single multiplier
            return ref_part, dial_part

    return token_clean, None


# ==========================================
# 5. DEKODER REFERENSI MULTI-MEREK
# ==========================================

import json
from pathlib import Path
from src.config import RULES_PATH as DYNAMIC_RULES_PATH


def get_dynamic_rules() -> Dict[str, Any]:
    """Membaca aturan dinamis hasil ekstraksi Gemini dari models/dynamic_rules.json."""
    if DYNAMIC_RULES_PATH.exists():
        try:
            return json.loads(DYNAMIC_RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def decode_reference(reference_str: str) -> Dict[str, Any]:
    """
    Mendekodekan nomor referensi multi-merek (VC, Rolex, Cartier, Tudor, Patek, AP).
    Mendukung aturan bawaan dan aturan dinamis hasil penemuan Gemini.
    """
    ref_clean = reference_str.strip().upper()
    result = {
        "brand": None,
        "series": None,
        "reference": ref_clean,
        "material": None,
        "dial": None,
    }

    # 0. Cek Dynamic Rules (Aturan baru hasil penemuan Gemini)
    dynamic_prefixes = get_dynamic_rules().get("prefix_map", {})
    for pfx, info in dynamic_prefixes.items():
        if ref_clean.startswith(pfx):
            result["brand"] = info.get("brand")
            result["series"] = info.get("series")
            result["material"] = info.get("material")
            return result

    # 1. Cek Cartier (Prefix WJPN, WGTA, WSSA, WSTA, WSBB, etc.)
    for pfx, (b, s, m) in CARTIER_PREFIX_MAP.items():
        if ref_clean.startswith(pfx):
            result["brand"] = b
            result["series"] = s
            result["material"] = m
            return result

    # 2. Cek Tudor (Prefix 2836, 7903, 7923, etc.)
    for pfx, (b, s, m) in TUDOR_PREFIX_MAP.items():
        if ref_clean.startswith(pfx):
            result["brand"] = b
            result["series"] = s
            result["material"] = m
            return result

    # 3. Cek Patek Philippe (Prefix 5711, 5712, 5330, 7300, etc.)
    for pfx, (b, s, m) in PATEK_PREFIX_MAP.items():
        if ref_clean.startswith(pfx):
            result["brand"] = b
            result["series"] = s
            result["material"] = m
            return result

    # 4. Cek Audemars Piguet (Prefix 15500, 15510, 16202, 26331, etc.)
    for pfx, (b, s, m) in AUDEMARS_PREFIX_MAP.items():
        if ref_clean.startswith(pfx):
            result["brand"] = b
            result["series"] = s
            result["material"] = m
            return result

    # 5. Cek Rolex (Prefix 1263, 1165, 1266, 1267, 3369, 2282, 5250, etc.)
    # Cek prefix 4-digit
    pfx_4 = ref_clean[:4]
    pfx_5 = ref_clean[:5]
    if pfx_4 in ROLEX_SERIES_PREFIX_MAP or pfx_5 in ROLEX_SERIES_PREFIX_MAP:
        result["brand"] = "Rolex"
        result["series"] = ROLEX_SERIES_PREFIX_MAP.get(pfx_4) or ROLEX_SERIES_PREFIX_MAP.get(pfx_5)

        # Cari digit terakhir dari 5 atau 6 angka pertama untuk material Rolex
        digits_match = re.match(r"^(\d{5,6})", ref_clean)
        if digits_match:
            last_digit = digits_match.group(1)[-1]
            result["material"] = ROLEX_MATERIAL_DIGIT.get(last_digit, "Stainless Steel")

        # Cek suffix bezel/dial khas Rolex (LV = Green, LN = Black, BLRO = Pepsi, etc.)
        if "LV" in ref_clean:
            result["dial"] = "Green Bezel / Black Dial"
        elif "BLRO" in ref_clean:
            result["dial"] = "Pepsi (Blue/Red Bezel)"
        elif "BLNR" in ref_clean:
            result["dial"] = "Batman (Blue/Black Bezel)"
        elif "CHNR" in ref_clean:
            result["dial"] = "Root Beer"

        return result

    # 6. Cek Vacheron Constantin (Prefix 4500V, 5500V, 82035, etc.)
    base_prefix = ref_clean.split("/")[0].split("-")[0].strip()
    if base_prefix in VC_SERIES_PREFIX_MAP or base_prefix[:5] in VC_SERIES_PREFIX_MAP:
        result["brand"] = "Vacheron Constantin"
        result["series"] = VC_SERIES_PREFIX_MAP.get(base_prefix) or VC_SERIES_PREFIX_MAP.get(base_prefix[:5])

        # Ekstraksi kode Material dari bagian tengah (misal /110A- atau /000R- atau -200R-)
        mat_match = re.search(r"[/|-](\d{3})([A-Z])", ref_clean)
        if mat_match:
            mat_code = mat_match.group(2)
            result["material"] = VC_MATERIAL_MAP.get(mat_code, mat_code)

        # Ekstraksi kode Suffix Dial (misal -B128, -B483, -B952)
        suffix_match = re.search(r"-([A-Z0-9]{4})$", ref_clean)
        if suffix_match:
            suffix = suffix_match.group(1)
            if suffix in VC_DIAL_MAP:
                result["dial"] = VC_DIAL_MAP[suffix]

        return result

    return result


def extract_explicit_dial(text: str) -> Optional[str]:
    """Mencari penyebutan warna dial eksplisit dalam teks baris broadcast."""
    text_lower = text.lower()
    for pattern, dial_name in EXPLICIT_DIAL_PATTERNS:
        if re.search(pattern, text_lower):
            return dial_name
    return None


def parse_price(price_str: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Mengubah teks harga menjadi (currency, numeric_amount).
    Mendukung format umum dealer WhatsApp:
    - 'hkd82500' -> ('HKD', 82500.0)
    - 'hld219000' -> ('HKD', 219000.0)  (typo HLD -> HKD)
    - 'HKD182K' -> ('HKD', 182000.0)
    - '445000HKHKD' -> ('HKD', 445000.0)
    - '$437000' -> ('HKD', 437000.0)
    - '169000HKD' -> ('HKD', 169000.0)
    """
    clean_p = price_str.upper().strip()

    # Perbaiki typo umum dealer
    clean_p = clean_p.replace("HLD", "HKD").replace("HKHKD", "HKD")

    # Deteksi currency
    currency = "HKD"  # default pasar dealer Hong Kong
    cur_match = re.search(r"(HKD|USD|SGD|IDR|EUR|GBP|RMB|CNY)", clean_p)
    if cur_match:
        currency = cur_match.group(1)

    # Bersihkan simbol mata uang dan karakter non-angka kecuali desimal & multiplier K/M
    num_part = re.sub(r"(HKD|USD|SGD|IDR|EUR|GBP|RMB|CNY|\$|/|\s|,)", "", clean_p).strip()

    multiplier = 1.0
    if "K" in num_part or "k" in num_part:
        multiplier = 1000.0
        num_part = re.sub(r"[Kk]", "", num_part).strip()
    elif "M" in num_part or "m" in num_part:
        multiplier = 1000000.0
        num_part = re.sub(r"[Mm]", "", num_part).strip()

    # Ambil angka pertama jika ada trailing karakter
    num_match = re.search(r"^([0-9]+(?:\.[0-9]+)?)", num_part)
    if num_match:
        try:
            val = float(num_match.group(1)) * multiplier
            return currency, val
        except ValueError:
            pass

    return currency, None

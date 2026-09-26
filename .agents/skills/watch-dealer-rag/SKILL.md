---
name: watch-dealer-rag
description: Fast CPU-based Watch Dealer NER and RAG Search Engine. Use this skill whenever the user wants to search watch dealer prices, inventory, references, dial colors (blue, green, silver), materials, or parse and index new dealer broadcast messages.
---

# Watch Dealer NER & RAG Search Skill

This skill allows Antigravity to search luxury watch inventories, extract structured data from dealer broadcast text, and query the local DuckDB RAG database at CPU speeds (< 100ms).

---

## 🔍 Search Commands

Execute search queries using `app.py`:

```bash
uv run app.py --search "<QUERY>" [OPTIONS]
```

### Options:
- `--max-price <HKD>`: Filter maximum price (e.g. `250000`).
- `--min-year <YEAR>`: Filter minimum year (e.g. `2022`).
- `--dial <COLOR>`: Filter dial color (e.g. `Blue`, `Green`, `Silver`, `Black`, `Salmon`).
- `--limit <N>`: Number of results (default: 5).

### Example Searches:
```bash
# 1. Search Overseas blue dial with price limit
uv run app.py --search "Overseas blue dial" --max-price 250000

# 2. Search Rose Gold with Green Dial
uv run app.py --search "rose gold green dial"

# 3. Search Historiques 222 or 1921
uv run app.py --search "Historiques"
```

---

## 📥 Ingesting & Parsing New Dealer Broadcasts

When the user provides new raw broadcast messages:
1. Save the text to a file in [datasets/](file:///home/anonym/Desktop/AI/datasets) or append to `datasets/raw_broadcasts.txt`.
2. Run indexing:
   ```bash
   uv run app.py --parse datasets/raw_broadcasts.txt
   ```
3. Retrain the custom spaCy NER model if new reference patterns are introduced:
   ```bash
   uv run src/trainer.py
   ```

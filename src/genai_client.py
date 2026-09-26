"""
src/genai_client.py - Modul Klien Resmi Google GenAI (Gemini 3.8 Flash).
Mengatur pembacaan API Key dari .env dan inisialisasi Client.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from google import genai

# Hilangkan warning internal SDK Google GenAI terkait Automatic Function Calling (AFC)
logging.getLogger("google_genai").setLevel(logging.ERROR)


from src.config import GEMINI_MODEL, ENV_FILE, load_env_file

DEFAULT_MODEL = GEMINI_MODEL


def load_env_if_needed():
    """Memastikan GEMINI_API_KEY terbaca dari .env jika belum ada di environment."""
    load_env_file()


def get_gemini_client() -> genai.Client:
    """Menginisialisasi dan mengembalikan instance genai.Client resmi."""
    load_env_if_needed()
    return genai.Client()


def generate_structured_content(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """Memanggil Gemini untuk menghasilkan konten teks atau JSON."""
    client = get_gemini_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text or ""

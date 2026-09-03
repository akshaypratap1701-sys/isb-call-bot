"""
ISB Call Bot — Configuration (Sarvam AI Only)
===============================================
No OpenAI dependency. Uses Sarvam AI for STT, LLM, and TTS.
"""

import os
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# ─── Paths ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
AUDIO_DIR = BASE_DIR / "app" / "audio_cache"

# ─── Sarvam AI ───────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_API_BASE = os.getenv("SARVAM_API_BASE", "https://api.sarvam.ai")

# ─── Sarvam Models ───────────────────────────────────────
SARVAM_LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
SARVAM_STT_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v3")
SARVAM_STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE", "en-IN")
SARVAM_STT_MODE = os.getenv("SARVAM_STT_MODE", "transcribe")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "priya")
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
SARVAM_TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))
SARVAM_TTS_FORMAT = os.getenv("SARVAM_TTS_FORMAT", "wav")

# ─── Twilio ──────────────────────────────────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# ─── Voice / Language ────────────────────────────────────
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-IN")
SPEECH_HINTS = "ISB, PGP, PGPMAX, MFAB, PGPpro, GMAT, GRE, Mohali, Hyderabad, Gachibowli, fellowship, admission"

# ─── Conversation ────────────────────────────────────────
MAX_TURNS = int(os.getenv("MAX_TURNS", "15"))
GATHER_TIMEOUT = int(os.getenv("GATHER_TIMEOUT", "10"))
MAX_SPEECH_DURATION = int(os.getenv("MAX_SPEECH_DURATION", "15"))

# ─── Server ──────────────────────────────────────────────
PORT = int(os.getenv("PORT", "8000"))
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "")

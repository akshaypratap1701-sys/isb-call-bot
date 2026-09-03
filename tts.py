"""
ISB Call Bot — Sarvam TTS Service
Generates speech audio using Sarvam Bulbul v3.
Caches audio files to avoid regenerating the same text twice.
"""

import os
import hashlib
import logging
import requests
from typing import Optional

from .config import (
    SARVAM_API_KEY, SARVAM_API_BASE,
    SARVAM_TTS_MODEL, SARVAM_TTS_SPEAKER,
    SARVAM_TTS_LANGUAGE, SARVAM_TTS_PACE, SARVAM_TTS_FORMAT,
    AUDIO_DIR,
)

logger = logging.getLogger(__name__)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(text: str) -> str:
    raw = f"{text}_{SARVAM_TTS_SPEAKER}_{SARVAM_TTS_LANGUAGE}_{SARVAM_TTS_PACE}"
    return hashlib.md5(raw.encode()).hexdigest()


def generate_speech(text: str) -> Optional[str]:
    if not SARVAM_API_KEY:
        logger.error("SARVAM_API_KEY not set")
        return None

    text = text[:2500]
    cache_key = _cache_key(text)
    audio_file = AUDIO_DIR / f"{cache_key}.{SARVAM_TTS_FORMAT}"

    if audio_file.exists():
        logger.info(f"TTS cache hit: {cache_key}")
        return str(audio_file)

    url = f"{SARVAM_API_BASE}/text-to-speech"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": [text],
        "target_language_code": SARVAM_TTS_LANGUAGE,
        "speaker": SARVAM_TTS_SPEAKER,
        "pace": SARVAM_TTS_PACE,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": SARVAM_TTS_MODEL,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "audios" in data and len(data["audios"]) > 0:
            import base64
            audio_bytes = base64.b64decode(data["audios"][0])
            with open(audio_file, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"TTS generated: {cache_key} ({len(audio_bytes)} bytes)")
            return str(audio_file)
        else:
            logger.error(f"Sarvam TTS unexpected response: {data}")
            return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None


def generate_speech_url(text: str, base_url: str) -> Optional[str]:
    audio_path = generate_speech(text)
    if audio_path is None:
        return None
    filename = os.path.basename(audio_path)
    return f"{base_url}/audio/{filename}"

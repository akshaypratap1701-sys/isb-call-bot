"""
ISB Call Bot — Sarvam STT Service
Transcribes speech audio using Sarvam Saaras v3.
"""

import logging
import requests
from typing import Optional

from .config import (
    SARVAM_API_KEY, SARVAM_API_BASE,
    SARVAM_STT_MODEL, SARVAM_STT_LANGUAGE, SARVAM_STT_MODE,
)

logger = logging.getLogger(__name__)


def transcribe_audio(audio_url: str) -> Optional[str]:
    if not SARVAM_API_KEY:
        return None
    url = f"{SARVAM_API_BASE}/speech-to-text"
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "model": SARVAM_STT_MODEL,
        "mode": SARVAM_STT_MODE,
        "language_code": SARVAM_STT_LANGUAGE,
        "url": audio_url,
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        if "transcript" in data:
            return data["transcript"].strip()
        return None
    except Exception as e:
        logger.error(f"STT error: {e}")
        return None

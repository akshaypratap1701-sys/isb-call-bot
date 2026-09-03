# ISB Call Bot (Sarvam AI)

A production-ready voice telephony bot for the Indian School of Business (ISB), powered entirely by Sarvam AI.

## Sarvam AI Stack

| Component | Model | Description |
|---|---|---|
| LLM | sarvam-105b-conversations | Tuned for voice agents and real-time dialogue |
| TTS | bulbul:v3 | Natural Indian voices, speaker: priya |
| STT | saaras:v3 | 22 Indian languages + English, telephony-optimized |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Fill in your keys
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
isb-call-bot/
├── app/
│   ├── __init__.py
│   ├── config.py            # Configuration
│   ├── main.py              # FastAPI + Twilio webhooks
│   ├── conversation.py      # Sarvam LLM + RAG
│   ├── tts.py               # Sarvam Bulbul v3 TTS
│   ├── stt.py               # Sarvam Saaras v3 STT
│   ├── knowledge_base.py    # ISB knowledge + TF-IDF retrieval
│   └── audio_cache/         # Cached TTS audio
├── data/
│   └── knowledge_base.json  # 22 ISB info entries
├── requirements.txt
├── Procfile
└── .env.example
```

See DEPLOYMENT.md for full deployment instructions.

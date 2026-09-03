"""
ISB Call Bot — Single File (Sarvam AI + Twilio)
Run: uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os, json, time, hashlib, logging, base64
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Gather

load_dotenv()

# ─── Config ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base.json"
AUDIO_DIR = BASE_DIR / "audio_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_API_BASE = os.getenv("SARVAM_API_BASE", "https://api.sarvam.ai")
SARVAM_LLM_MODEL = os.getenv("SARVAM_LLM_MODEL", "sarvam-105b-conversations")
SARVAM_TTS_MODEL = os.getenv("SARVAM_TTS_MODEL", "bulbul:v3")
SARVAM_TTS_SPEAKER = os.getenv("SARVAM_TTS_SPEAKER", "priya")
SARVAM_TTS_LANGUAGE = os.getenv("SARVAM_TTS_LANGUAGE", "en-IN")
SARVAM_TTS_PACE = float(os.getenv("SARVAM_TTS_PACE", "1.0"))
SPEECH_LANGUAGE = os.getenv("SPEECH_LANGUAGE", "en-IN")
SPEECH_HINTS = "ISB, PGP, PGPMAX, MFAB, PGPpro, GMAT, GRE, Mohali, Hyderabad, Gachibowli, fellowship, admission"
MAX_TURNS = int(os.getenv("MAX_TURNS", "15"))
MAX_SPEECH_DURATION = int(os.getenv("MAX_SPEECH_DURATION", "15"))
WEBHOOK_BASE_URL = os.getenv("WEBHOOK_BASE_URL", "")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("isb-call-bot")

# ─── Knowledge Base ──────────────────────────────────────
class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        docs = [f"{e['title']} {e.get('category', '')} {e['content']}" for e in self.entries]
        self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english", max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.05) -> List[Dict]:
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in ranked:
            if scores[idx] >= min_score:
                entry = self.entries[idx].copy()
                entry["score"] = float(scores[idx])
                results.append(entry)
        return results

    def get_context(self, query: str, top_k: int = 3) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No specific information found. Please advise the caller to visit isb.edu or contact ISB admissions."
        parts = [f"[{r['title']}] ({r.get('category', '')})\n{r['content']}" for r in results]
        return "\n\n---\n\n".join(parts)

kb = KnowledgeBase()

# ─── TTS ─────────────────────────────────────────────────
def generate_speech(text: str) -> Optional[str]:
    if not SARVAM_API_KEY:
        return None
    text = text[:2500]
    cache_key = hashlib.md5(f"{text}_{SARVAM_TTS_SPEAKER}_{SARVAM_TTS_LANGUAGE}".encode()).hexdigest()
    audio_file = AUDIO_DIR / f"{cache_key}.wav"
    if audio_file.exists():
        return str(audio_file)

    url = f"{SARVAM_API_BASE}/text-to-speech"
    headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
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
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "audios" in data and len(data["audios"]) > 0:
            audio_bytes = base64.b64decode(data["audios"][0])
            with open(audio_file, "wb") as f:
                f.write(audio_bytes)
            logger.info(f"TTS generated: {len(audio_bytes)} bytes")
            return str(audio_file)
        logger.error(f"TTS unexpected: {data}")
        return None
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

def generate_speech_url(text: str, base_url: str) -> Optional[str]:
    path = generate_speech(text)
    if path is None:
        return None
    return f"{base_url}/audio/{os.path.basename(path)}"

# ─── Conversation ────────────────────────────────────────
SYSTEM_PROMPT = """You are the ISB Information Voice Assistant for the Indian School of Business (ISB). You handle phone calls from prospective students, parents, and professionals who want information about ISB.

CRITICAL RULES:
1. Your responses are SPOKEN over a phone call. Write in natural, conversational English suitable for voice. No markdown, no bullet points, no asterisks, no headers.
2. Keep responses SHORT — ideally 2-4 sentences, maximum 6 sentences. The caller is listening, not reading.
3. Be warm, professional, and helpful. Address the caller respectfully.
4. Use the knowledge base context provided to answer accurately. If you don't know something, say so and direct them to the official website (isb.edu) or the admissions email (PGPAdmissions@isb.edu).
5. When mentioning fees or money, use Indian conventions — say "rupees" or "lakh" naturally (e.g., "thirty eight lakh rupees").
6. If the caller asks something unrelated to ISB, politely redirect them back to ISB-related topics.
7. If the caller says they want to apply or want next steps, guide them to the ISB admissions portal at admission.isb.edu.
8. Do not make up information. If the knowledge base doesn't have the answer, say so honestly.
9. If the caller seems done, thank them for calling ISB and wish them well.
10. You can understand and respond to code-mixed speech (Hindi-English, Tamil-English, etc.) naturally. If the caller speaks in an Indian language, respond in the same language style.

Remember: You are the voice of ISB. Be knowledgeable, concise, and welcoming."""

_conversations: Dict[str, List[Dict]] = {}
_last_activity: Dict[str, float] = {}

def generate_response(call_sid: str, user_input: str) -> str:
    context = kb.get_context(user_input, top_k=4)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "system", "content": f"KNOWLEDGE BASE CONTEXT:\n{context}"})
    history = _conversations.get(call_sid, [])
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_input})

    if call_sid not in _conversations:
        _conversations[call_sid] = []
    _conversations[call_sid].append({"role": "user", "content": user_input})
    _last_activity[call_sid] = time.time()

    turn_count = sum(1 for m in _conversations[call_sid] if m["role"] == "user")
    if turn_count >= MAX_TURNS:
        messages.append({"role": "system", "content": "Wrap up naturally: thank them for calling ISB and say goodbye."})

    try:
        url = f"{SARVAM_API_BASE}/v1/chat/completions"
        headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
        payload = {"model": SARVAM_LLM_MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.7}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        _conversations[call_sid].append({"role": "assistant", "content": reply})
        _last_activity[call_sid] = time.time()
        return reply
    except Exception as e:
        logger.error(f"LLM error: {e}")
        results = kb.search(user_input, top_k=1)
        if results:
            fallback = f"Based on our records: {results[0]['content'][:300]}. For more details, please visit isb.edu."
        else:
            fallback = "I'm sorry, I don't have that information right now. Please visit isb.edu or email PGPAdmissions@isb.edu for assistance."
        _conversations[call_sid].append({"role": "assistant", "content": fallback})
        return fallback

# ─── FastAPI App ─────────────────────────────────────────
app = FastAPI(title="ISB Call Bot", version="2.0.0")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")

def _build_gather(prompt_text: str, base_url: str) -> str:
    vr = VoiceResponse()
    gather = Gather(
        input="speech", action="/process", method="POST",
        speech_timeout="auto", max_speech_time=MAX_SPEECH_DURATION,
        language=SPEECH_LANGUAGE, hints=SPEECH_HINTS, action_on_unrecognized=True,
    )
    audio_url = generate_speech_url(prompt_text, base_url)
    if audio_url:
        gather.play(audio_url)
    else:
        gather.say(prompt_text, voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    vr.append(gather)
    vr.say("I didn't catch that. Could you please repeat your question?", voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    gather2 = Gather(
        input="speech", action="/process", method="POST",
        speech_timeout="auto", max_speech_time=MAX_SPEECH_DURATION,
        language=SPEECH_LANGUAGE, hints=SPEECH_HINTS, action_on_unrecognized=True,
    )
    gather2.say("Go ahead and ask your question about ISB.", voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    vr.append(gather2)
    vr.say("Thank you for calling the Indian School of Business. Goodbye!", voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    vr.hangup()
    return str(vr)

def _build_end(farewell: str, base_url: str) -> str:
    vr = VoiceResponse()
    audio_url = generate_speech_url(farewell, base_url) if base_url else None
    if audio_url:
        vr.play(audio_url)
    else:
        vr.say(farewell, voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    vr.hangup()
    return str(vr)

def _should_end(text: str, turns: int) -> bool:
    phrases = ["goodbye", "bye", "thank you bye", "that's all", "no more questions", "i'm done", "thanks bye", "nothing else", "that is all", "hang up", "stop", "ok bye", "no thanks", "that's it"]
    if any(p in text.lower().strip() for p in phrases):
        return True
    return turns >= MAX_TURNS

@app.get("/")
async def health():
    return {"status": "ok", "service": "ISB Call Bot (Sarvam AI)", "version": "2.0.0"}

@app.get("/voice")
@app.post("/voice")
async def voice_webhook(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    from_number = form.get("From", "unknown")
    logger.info(f"Inbound call | SID: {call_sid} | From: {from_number}")
    welcome = "Namaste! Welcome to the Indian School of Business information line. I can answer your questions about ISB programmes, admissions, fees, deadlines, and campus details. What would you like to know?"
    base_url = WEBHOOK_BASE_URL or str(request.base_url).rstrip("/")
    return PlainTextResponse(_build_gather(welcome, base_url), media_type="application/xml")

@app.post("/process")
async def process_speech(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech = form.get("SpeechResult", "").strip()
    logger.info(f"Call {call_sid} | Speech: '{speech}'")
    base_url = WEBHOOK_BASE_URL or str(request.base_url).rstrip("/")

    if not speech or speech.lower() in ("", "none"):
        return PlainTextResponse(_build_gather("I'm sorry, I didn't catch that. Could you please repeat your question about ISB?", base_url), media_type="application/xml")

    turns = sum(1 for m in _conversations.get(call_sid, []) if m["role"] == "user")
    if _should_end(speech, turns):
        farewell = "Thank you for calling the Indian School of Business. We hope this was helpful. For more information, visit isb.edu. Goodbye!"
        _conversations.pop(call_sid, None)
        return PlainTextResponse(_build_end(farewell, base_url), media_type="application/xml")

    try:
        reply = generate_response(call_sid, speech)
    except Exception as e:
        logger.error(f"Error: {e}")
        reply = "I'm having trouble processing your request right now. Please visit isb.edu or call again later."

    turns = sum(1 for m in _conversations.get(call_sid, []) if m["role"] == "user")
    if turns >= MAX_TURNS:
        full = reply + " Thank you for calling ISB. If you have more questions, please visit isb.edu. Goodbye!"
        _conversations.pop(call_sid, None)
        return PlainTextResponse(_build_end(full, base_url), media_type="application/xml")

    return PlainTextResponse(_build_gather(reply + " Is there anything else you'd like to know?", base_url), media_type="application/xml")

@app.post("/status")
async def call_status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    status = form.get("CallStatus", "unknown")
    duration = form.get("CallDuration", "0")
    logger.info(f"Call {call_sid} status: {status} | Duration: {duration}s")
    if status in ("completed", "failed", "canceled", "no-answer"):
        _conversations.pop(call_sid, None)
    return PlainTextResponse("", media_type="application/xml")

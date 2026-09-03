"""
ISB Call Bot — FastAPI Application (Sarvam AI + Twilio)
Fully Sarvam-powered voice bot. Twilio handles telephony and speech
recognition (via <Gather>), Sarvam handles LLM + TTS.

Run: uvicorn app.main:app --host 0.0.0.0 --port $PORT
"""

import logging
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from twilio.twiml.voice_response import VoiceResponse, Gather

from .config import (
    MAX_SPEECH_DURATION, SPEECH_LANGUAGE, SPEECH_HINTS,
    WEBHOOK_BASE_URL, PORT, MAX_TURNS, AUDIO_DIR,
)
from .conversation import ConversationManager
from .tts import generate_speech_url

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("isb-call-bot")

app = FastAPI(title="ISB Call Bot (Sarvam AI)", version="2.0.0")
conv = ConversationManager()
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


def _build_gather_response(prompt_text: str, base_url: str) -> str:
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
        logger.warning("Sarvam TTS failed, falling back to Polly")
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


def _build_end_response(farewell_text: str, base_url: str) -> str:
    vr = VoiceResponse()
    audio_url = generate_speech_url(farewell_text, base_url) if base_url else None
    if audio_url:
        vr.play(audio_url)
    else:
        vr.say(farewell_text, voice="Polly.Raveena", language=SPEECH_LANGUAGE)
    vr.hangup()
    return str(vr)


def _should_end_call(user_input: str, turn_count: int) -> bool:
    end_phrases = [
        "goodbye", "bye", "thank you bye", "that's all", "no more questions",
        "i'm done", "thanks bye", "nothing else", "that is all", "end the call",
        "hang up", "stop", "thank you goodbye", "ok bye", "no thanks", "that's it",
    ]
    lowered = user_input.lower().strip()
    if any(phrase in lowered for phrase in end_phrases):
        return True
    if turn_count >= MAX_TURNS:
        return True
    return False


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

    welcome = (
        "Namaste! Welcome to the Indian School of Business information line. "
        "I can answer your questions about ISB programmes, admissions, fees, "
        "deadlines, and campus details. What would you like to know?"
    )
    base_url = str(request.base_url).rstrip("/")
    if WEBHOOK_BASE_URL:
        base_url = WEBHOOK_BASE_URL
    return PlainTextResponse(_build_gather_response(welcome, base_url), media_type="application/xml")


@app.post("/process")
async def process_speech(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    speech_result = form.get("SpeechResult", "").strip()
    confidence = form.get("Confidence", "0")
    logger.info(f"Call {call_sid} | Speech: '{speech_result}' | Confidence: {confidence}")

    base_url = str(request.base_url).rstrip("/")
    if WEBHOOK_BASE_URL:
        base_url = WEBHOOK_BASE_URL

    if not speech_result or speech_result.lower() in ("", "none"):
        return PlainTextResponse(
            _build_gather_response("I'm sorry, I didn't catch that. Could you please repeat your question about ISB?", base_url),
            media_type="application/xml",
        )

    turn_count = conv.turn_count(call_sid)
    if _should_end_call(speech_result, turn_count):
        farewell = "Thank you for calling the Indian School of Business. We hope this was helpful. For more information, visit isb.edu. Goodbye!"
        twiml = _build_end_response(farewell, base_url)
        conv.end_conversation(call_sid)
        return PlainTextResponse(twiml, media_type="application/xml")

    try:
        bot_reply = conv.generate_response(call_sid, speech_result)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        bot_reply = "I'm having trouble processing your request right now. Please visit isb.edu or call again later."

    turn_count = conv.turn_count(call_sid)
    if turn_count >= MAX_TURNS:
        full_reply = bot_reply + " Thank you for calling ISB. If you have more questions, please visit isb.edu or email PGPAdmissions at isb.edu. Goodbye!"
        twiml = _build_end_response(full_reply, base_url)
        conv.end_conversation(call_sid)
        return PlainTextResponse(twiml, media_type="application/xml")

    prompt = bot_reply + " Is there anything else you'd like to know?"
    return PlainTextResponse(_build_gather_response(prompt, base_url), media_type="application/xml")


@app.post("/status")
async def call_status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "unknown")
    call_status = form.get("CallStatus", "unknown")
    duration = form.get("CallDuration", "0")
    logger.info(f"Call {call_sid} status: {call_status} | Duration: {duration}s")
    if call_status in ("completed", "failed", "canceled", "no-answer"):
        conv.end_conversation(call_sid)
    return PlainTextResponse("", media_type="application/xml")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False, log_level="info")

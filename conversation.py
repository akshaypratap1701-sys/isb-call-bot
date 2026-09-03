"""
ISB Call Bot — Conversation Manager (Sarvam AI)
Uses Sarvam-105B-Conversations LLM with RAG knowledge base.
"""

import time
import logging
from typing import Dict, List, Optional
import requests

from .config import SARVAM_API_KEY, SARVAM_API_BASE, SARVAM_LLM_MODEL, MAX_TURNS
from .knowledge_base import get_knowledge_base

logger = logging.getLogger(__name__)

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

CONVERSATION_TTL = 3600
_conversations: Dict[str, List[Dict]] = {}
_last_activity: Dict[str, float] = {}


class ConversationManager:
    def __init__(self):
        self.kb = get_knowledge_base()

    def _cleanup_old(self):
        now = time.time()
        expired = [sid for sid, ts in _last_activity.items() if now - ts > CONVERSATION_TTL]
        for sid in expired:
            _conversations.pop(sid, None)
            _last_activity.pop(sid, None)

    def get_history(self, call_sid: str) -> List[Dict]:
        self._cleanup_old()
        return _conversations.get(call_sid, [])

    def turn_count(self, call_sid: str) -> int:
        history = _conversations.get(call_sid, [])
        return sum(1 for m in history if m["role"] == "user")

    def generate_response(self, call_sid: str, user_input: str) -> str:
        self._cleanup_old()
        context = self.kb.get_context(user_input, top_k=4)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "system", "content": f"KNOWLEDGE BASE CONTEXT (use this to answer the caller's question):\n{context}"})
        history = self.get_history(call_sid)
        messages.extend(history[-10:])
        messages.append({"role": "user", "content": user_input})

        if call_sid not in _conversations:
            _conversations[call_sid] = []
        _conversations[call_sid].append({"role": "user", "content": user_input})
        _last_activity[call_sid] = time.time()

        if self.turn_count(call_sid) >= MAX_TURNS:
            messages.append({"role": "system", "content": "The caller has asked several questions. Wrap up naturally: offer any final help, thank them for calling ISB, and say goodbye."})

        if not SARVAM_API_KEY:
            fallback = self._fallback_response(user_input, context)
            _conversations[call_sid].append({"role": "assistant", "content": fallback})
            return fallback

        try:
            url = f"{SARVAM_API_BASE}/v1/chat/completions"
            headers = {"api-subscription-key": SARVAM_API_KEY, "Content-Type": "application/json"}
            payload = {"model": SARVAM_LLM_MODEL, "messages": messages, "max_tokens": 300, "temperature": 0.7}
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"].strip()
            _conversations[call_sid].append({"role": "assistant", "content": reply})
            _last_activity[call_sid] = time.time()
            return reply
        except Exception as e:
            logger.error(f"Sarvam LLM API error: {e}")
            fallback = self._fallback_response(user_input, context)
            _conversations[call_sid].append({"role": "assistant", "content": fallback})
            return fallback

    def _fallback_response(self, query: str, context: str) -> str:
        results = self.kb.search(query, top_k=1)
        if results:
            entry = results[0]
            return f"Based on our records: {entry['content'][:300]}. For more details, please visit isb.edu or call us again."
        return "I'm sorry, I don't have that information right now. Please visit isb.edu or email PGPAdmissions@isb.edu for assistance."

    def end_conversation(self, call_sid: str):
        _conversations.pop(call_sid, None)
        _last_activity.pop(call_sid, None)

"""LLM integration supporting offline Ollama, online APIs, and local fallback."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

from src.knowledge_graph import ReasoningResult


ITALIAN_HINTS = (
    "ciao",
    "grazie",
    "pioggia",
    "annullat",
    "partita",
    "triste",
    "ansia",
    "stress",
    "pericolo",
    "aiuto",
    "fuoco",
    "fumo",
    "bravo",
    "ottimo",
    "stupido",
    "arrabbi",
    "odio",
    "deluso",
    "male",
    "bene",
    "oggi",
    "sono",
    "non ",
    "perché",
    "perche",
    "mia ",
    "mio ",
    "madre",
    "padre",
    "tagliat",
    "cavi",
    "rotto",
    "spaccat",
    "distrutt",
    "ho ",
    "preso",
    "brutto",
    "butto",
    "voto",
    "esame",
)


def _looks_italian(text: Optional[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(token in lower for token in ITALIAN_HINTS)


class OllamaClient:
    """Unified LLM client for offline Ollama, online Gemini/OpenAI, and fallback."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 15.0,
        mode: str = "auto",
        online_provider: str = "gemini",
        online_model: str = "",
        api_key: Optional[str] = None,
        online_base_url: str = "https://api.openai.com/v1",
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").replace("://localhost", "://127.0.0.1")
        self.timeout = timeout
        self.mode = mode  # "auto", "offline", "online", "fallback"
        self.online_provider = online_provider  # "gemini" or "openai"
        self.online_model = online_model
        self.api_key = api_key
        self.online_base_url = online_base_url.rstrip("/")
        self.last_source: str = "fallback"
        self.last_error: Optional[str] = None

    def _resolve_api_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key.strip()
        if self.online_provider == "gemini":
            return os.environ.get("GEMINI_API_KEY")
        return os.environ.get("OPENAI_API_KEY")

    def _call_ollama(self, prompt: str) -> Optional[str]:
        base = self.base_url.replace("://localhost", "://127.0.0.1")
        url = f"{base}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(url, json=payload, timeout=(0.45, self.timeout))
            response.raise_for_status()
            data = response.json()
            text = str(data.get("response", "")).strip()
            return text or None
        except (requests.RequestException, ValueError) as exc:
            self.last_error = f"Ollama non raggiungibile ({exc.__class__.__name__})"
            return None

    def _call_gemini(self, prompt: str, api_key: str) -> Optional[str]:
        model = self.online_model or "gemini-2.0-flash"
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={api_key}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(str(p.get("text", "")) for p in parts).strip()
            return text or None
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None

    def _call_openai(self, prompt: str, api_key: str) -> Optional[str]:
        model = self.online_model or "gpt-4o-mini"
        url = f"{self.online_base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
        }
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            if not choices:
                return None
            text = str(choices[0].get("message", {}).get("content", "")).strip()
            return text or None
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return None

    def _call_online(self, prompt: str) -> Optional[str]:
        api_key = self._resolve_api_key()
        if not api_key:
            self.last_error = "API key mancante per modalità online"
            return None
        if self.online_provider == "openai":
            return self._call_openai(prompt, api_key)
        return self._call_gemini(prompt, api_key)

    def _complete(self, prompt: str) -> Tuple[Optional[str], str]:
        """Run prompt through the selected backend(s), returning (text, source)."""
        self.last_error = None
        mode = self.mode.lower()
        if mode == "fallback":
            return None, "fallback"

        if mode == "online":
            text = self._call_online(prompt)
            if text:
                return text, f"online:{self.online_provider}"
            return None, "fallback"

        if mode == "offline":
            text = self._call_ollama(prompt)
            if text:
                return text, f"offline:{self.model}"
            return None, "fallback"

        # Auto mode: online (if key present) -> offline Ollama -> fallback
        if self._resolve_api_key():
            text = self._call_online(prompt)
            if text:
                return text, f"online:{self.online_provider}"

        text = self._call_ollama(prompt)
        if text:
            return text, f"offline:{self.model}"

        return None, "fallback"

    def _build_extraction_prompt(
        self,
        text: str,
        valid_events: List[str],
        valid_contexts: List[str],
    ) -> str:
        return (
            "You classify a free-text event for a social robot ontology.\n"
            f"Valid events: {valid_events}\n"
            f"Valid contexts: {valid_contexts}\n"
            "Map disappointing external events (e.g. cancelled match, bad news, failure) "
            "to 'task_failure' or 'user_distress'.\n"
            "Map insults or aggression to 'user_hostility'.\n"
            "Map danger/fire/smoke/emergencies to 'safety_alert'.\n"
            'Return ONLY valid JSON: {"event": "<event>", "context": "<context>"}.\n\n'
            f'Text: "{text}"\nJSON:'
        )

    def _fallback_extract_event(
        self,
        text: str,
        valid_events: List[str],
        valid_contexts: List[str],
    ) -> Tuple[str, str]:
        """Rule-based IT/EN event extractor when no LLM is reachable."""
        lower = text.lower().strip()
        if not lower:
            return "user_silence", "casual_conversation"

        rules: List[Tuple[Tuple[str, ...], str, str]] = [
            (
                ("fuoco", "fumo", "incendio", "pericolo", "emergenza", "incidente", "ospedale", "sangue", "smoke", "fire", "danger", "emergency", "help!", "allarme"),
                "safety_alert",
                "emergency",
            ),
            (
                ("stupido", "idiota", "inutile", "odio", "vaff", "schifo", "arrabbi", "incazz", "furios", "rabbia", "nervos", "litigat", "tagliat", "spaccat", "distrutt", "rubat", "sequestrat", "tradit", "ingiust", "urlat", "insult", "stupid", "useless", "shut up", "hate", "angry", "furious", "destroyed", "stole", "cut my"),
                "user_hostility",
                "casual_conversation",
            ),
            (
                ("annullat", "rimandat", "pioggia", "fallit", "perso", "rotto", "rovinat", "guast", "errore", "bug", "bocciat", "brutto voto", "butto voto", "brutto", "insuffic", "peccato", "multa", "ritardo", "cancelled", "canceled", "failed", "rain", "broken", "lost", "ruined", "bad grade", "failed exam"),
                "task_failure",
                "casual_conversation",
            ),
            (
                ("triste", "ansia", "stress", "male", "piang", "solo", "stanco", "preoccupat", "sopraffatt", "depress", "soffr", "paura", "sad", "anxious", "stressed", "overwhelmed", "depressed", "worried", "cry"),
                "user_distress",
                "emotional_support",
            ),
            (
                ("critica", "sbagliato", "deluso", "lento", "pessimo", "wrong", "disappointed", "bad job"),
                "user_criticism",
                "assistance",
            ),
            (
                ("bravo", "grazie", "grande", "fantastico", "ottimo", "perfetto", "complimenti", "adoro", "thanks", "thank you", "awesome", "great job", "amazing"),
                "user_praise",
                "casual_conversation",
            ),
            (
                ("riuscito", "risolto", "vinto", "funziona", "fatto", "promosso", "laureat", "felice", "content", "30", "solved", "success", "worked", "won", "done", "happy"),
                "task_success",
                "assistance",
            ),
            (
                ("ciao", "buongiorno", "buonasera", "salve", "ehi", "hello", "hi ", "hey", "good morning"),
                "user_greeting",
                "casual_conversation",
            ),
        ]

        for keywords, event, context in rules:
            if any(kw in lower for kw in keywords):
                ev = event if event in valid_events else valid_events[0]
                ctx = context if context in valid_contexts else valid_contexts[0]
                return ev, ctx

        # Never default unknown text to a positive greeting; fall back to neutral pause
        neutral_ev = "user_silence" if "user_silence" in valid_events else valid_events[0]
        neutral_ctx = "casual_conversation" if "casual_conversation" in valid_contexts else valid_contexts[0]
        return neutral_ev, neutral_ctx

    def extract_event(
        self,
        text: str,
        valid_events: List[str],
        valid_contexts: List[str],
        preferred_context: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Extract (event, context, source) from free text using LLM or fallback."""
        prompt = self._build_extraction_prompt(text, valid_events, valid_contexts)
        raw, source = self._complete(prompt)

        if raw:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    ev = str(parsed.get("event", "")).strip()
                    ctx = str(parsed.get("context", "")).strip()
                    if ev in valid_events:
                        final_ctx = (
                            preferred_context
                            if preferred_context in valid_contexts
                            else (ctx if ctx in valid_contexts else "casual_conversation")
                        )
                        return ev, final_ctx, source
                except ValueError:
                    pass

        ev, ctx = self._fallback_extract_event(text, valid_events, valid_contexts)
        if preferred_context and preferred_context in valid_contexts:
            ctx = preferred_context
        return ev, ctx, "fallback"

    def _build_prompt(
        self,
        reasoning: ReasoningResult,
        user_utterance: Optional[str] = None,
    ) -> str:
        """Build a grounded prompt from the Knowledge Graph reasoning trace."""
        trace_block = "\n".join(f"- {step}" for step in reasoning.reasoning_trace)
        utterance_line = (
            f'User utterance: "{user_utterance}"\n' if user_utterance else ""
        )

        return (
            "You are a social robot. Generate a short, natural, context-aware verbal response "
            "(1 to 3 sentences) in the same language as the user utterance, strictly grounded "
            "in the Knowledge Graph decision below. Do not contradict the selected behavior.\n\n"
            f"{utterance_line}"
            f"Interaction context: {reasoning.context}\n"
            f"Perceived event: {reasoning.event}\n"
            f"Robot internal emotion: {reasoning.new_emotion} (intensity: {reasoning.intensity.value})\n"
            f"Selected behavior: {reasoning.selected_behavior.name}\n"
            f"Required tone: {reasoning.selected_behavior.tone}\n"
            f"Reasoning trace:\n{trace_block}\n\n"
            "Robot response:"
        )

    def _fallback_response(
        self,
        reasoning: ReasoningResult,
        user_utterance: Optional[str] = None,
    ) -> str:
        """Deterministic IT/EN template fallback when no LLM is available."""
        if _looks_italian(user_utterance):
            it_templates: Dict[str, str] = {
                "warm_greeting": "Ciao! Che piacere sentirti. Come posso aiutarti oggi?",
                "enthusiastic_acknowledgment": "Fantastico, mi fa davvero piacere! Sono contento che sia andata bene.",
                "empathetic_listening": "Capisco come ti senti e mi dispiace molto. Sono qui con te se vuoi parlarne.",
                "calm_reassurance": "Stai tranquillo, affrontiamo tutto con calma un passo alla volta.",
                "polite_apology": f"Che peccato per '{user_utterance}', mi dispiace davvero. Possiamo trovare subito un'alternativa.",
                "proactive_check_in": "Sono qui quando vuoi. C'è qualcosa in cui posso darti una mano?",
                "urgent_safety_guidance": "Attenzione: fermiamoci subito e seguiamo le procedure di sicurezza!",
                "firm_boundary": "Preferisco mantenere un tono rispettoso. Quando vuoi, possiamo parlarne con calma.",
            }
            return it_templates.get(
                reasoning.selected_behavior.name,
                "Sono qui per aiutarti.",
            )

        en_templates: Dict[str, str] = {
            "warm_greeting": "Hello! It is great to see you. How can I help you today?",
            "enthusiastic_acknowledgment": "Thank you! I am really glad that went well.",
            "empathetic_listening": "I hear you, and I am here with you. Take your time if you want to share more.",
            "calm_reassurance": "Everything is under control. Let us take this one step at a time.",
            "polite_apology": "I am sorry about that. Let me adjust and help you find an alternative right away.",
            "proactive_check_in": "I am here whenever you are ready. Would you like assistance with anything?",
            "urgent_safety_guidance": "Attention: please pause and follow safety procedures immediately.",
            "firm_boundary": "I want to help, but let us keep our conversation respectful.",
        }
        return en_templates.get(
            reasoning.selected_behavior.name,
            "I am here to assist you.",
        )

    def generate_response(
        self,
        reasoning: ReasoningResult,
        user_utterance: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Generate a verbal response via configured LLM, returning (text, used_llm)."""
        prompt = self._build_prompt(reasoning, user_utterance)
        text, source = self._complete(prompt)
        self.last_source = source
        if text:
            return text, True

        self.last_source = "fallback"
        return self._fallback_response(reasoning, user_utterance), False

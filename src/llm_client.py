"""LLM integration supporting offline Ollama, online APIs, and local fallback."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests

from src.knowledge_graph import ReasoningResult


ITALIAN_STOPWORDS = {
    "ciao", "buongiorno", "buonasera", "salve", "grazie", "prego", "scusa",
    "sono", "sei", "siamo", "ho", "hai", "ha", "abbiamo", "hanno",
    "mi", "ti", "ci", "vi", "si", "che", "chi", "cosa", "come", "perche",
    "perché", "oggi", "ieri", "domani", "ora", "adesso", "sempre", "mai",
    "molto", "troppo", "male", "bene", "tutto", "niente", "nulla",
    "casa", "mamma", "madre", "padre", "papa", "papà", "famiglia", "amico",
    "amica", "lavoro", "esame", "voto", "partita", "pioggia", "cavi",
    "non", "mia", "mio", "tuo", "tua", "suo", "sua", "con", "per", "una",
    "uno", "del", "della", "dei", "delle", "nel", "nella", "sul", "sulla",
    "preso", "perso", "fatto", "rotto", "cacciato", "tagliato", "annullata",
    "triste", "felice", "contento", "contenta", "arrabbiato", "arrabbiata",
    "paura", "ansia", "stress", "aiuto", "pericolo", "fumo", "fuoco",
}


def _looks_italian(text: Optional[str]) -> bool:
    """Detect whether free text is likely written in Italian."""
    if not text:
        return False
    lower = text.lower()
    if any(ch in lower for ch in ("à", "è", "é", "ì", "ò", "ù")):
        return True
    tokens = set(re.findall(r"[a-zàèéìòù]+", lower))
    return bool(tokens & ITALIAN_STOPWORDS)


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
        if self.online_provider == "groq":
            return os.environ.get("GROQ_API_KEY")
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

    def _record_http_error(self, exc: requests.HTTPError) -> None:
        """Extract provider status code and error message into last_error."""
        detail = ""
        if exc.response is not None:
            try:
                err_json = exc.response.json()
                detail = str(err_json.get("error", {}).get("message", "")).strip()
            except ValueError:
                detail = exc.response.text[:120].strip()
            self.last_error = f"HTTP {exc.response.status_code}: {detail or exc}"
        else:
            self.last_error = str(exc)

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
        except requests.HTTPError as exc:
            self._record_http_error(exc)
            return None
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            self.last_error = f"Errore Gemini ({exc.__class__.__name__})"
            return None

    def _call_openai(
        self,
        prompt: str,
        api_key: str,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
    ) -> Optional[str]:
        model = self.online_model or default_model
        url = f"{(base_url or self.online_base_url).rstrip('/')}/chat/completions"
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
        except requests.HTTPError as exc:
            self._record_http_error(exc)
            return None
        except (requests.RequestException, ValueError, KeyError, IndexError) as exc:
            self.last_error = f"Errore provider online ({exc.__class__.__name__})"
            return None

    def _call_online(self, prompt: str) -> Optional[Tuple[str, str]]:
        api_key = self._resolve_api_key()
        provider = self.online_provider

        if not api_key and not self.api_key:
            for env_name, env_provider in (
                ("GEMINI_API_KEY", "gemini"),
                ("GROQ_API_KEY", "groq"),
                ("OPENAI_API_KEY", "openai"),
            ):
                val = os.environ.get(env_name)
                if val:
                    api_key = val.strip()
                    provider = env_provider
                    break

        if not api_key:
            self.last_error = "API key mancante per modalità online"
            return None
        if provider == "groq":
            text = self._call_openai(
                prompt,
                api_key,
                base_url="https://api.groq.com/openai/v1",
                default_model="openai/gpt-oss-20b",
            )
            return (text, "online:groq") if text else None
        if provider == "openai":
            text = self._call_openai(prompt, api_key)
            return (text, "online:openai") if text else None
        text = self._call_gemini(prompt, api_key)
        return (text, "online:gemini") if text else None

    def _complete(self, prompt: str) -> Tuple[Optional[str], str]:
        """Run prompt through the selected backend(s), returning (text, source)."""
        self.last_error = None
        mode = self.mode.lower()
        if mode == "fallback":
            return None, "fallback"

        if mode == "online":
            res = self._call_online(prompt)
            if res:
                return res
            return None, "fallback"

        if mode == "offline":
            text = self._call_ollama(prompt)
            if text:
                return text, f"offline:{self.model}"
            return None, "fallback"

        # Auto mode: online (if key present) -> offline Ollama -> fallback
        online_error: Optional[str] = None
        if self._resolve_api_key() or any(
            os.environ.get(k) for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY")
        ):
            res = self._call_online(prompt)
            if res:
                return res
            online_error = self.last_error

        text = self._call_ollama(prompt)
        if text:
            return text, f"offline:{self.model}"

        if online_error and self.last_error and online_error != self.last_error:
            self.last_error = f"{online_error} | {self.last_error}"
        elif online_error and not self.last_error:
            self.last_error = online_error

        return None, "fallback"

    def _build_extraction_prompt(
        self,
        text: str,
        valid_events: List[str],
        valid_contexts: List[str],
    ) -> str:
        return (
            "You classify a free-text social interaction event for a robot ontology.\n"
            f"Valid events: {valid_events}\n"
            f"Valid contexts: {valid_contexts}\n"
            "Classification guidelines:\n"
            "- Personal/family crisis, grief, loneliness, fear, anxiety, being kicked out -> "
            "'user_distress' in 'emotional_support'.\n"
            "- Disappointing setbacks (cancelled match, failed exam, lost items, broken device) -> "
            "'task_failure' in 'casual_conversation' or 'assistance'.\n"
            "- Insults, aggression, theft, or sabotage -> 'user_hostility' in 'casual_conversation'.\n"
            "- Criticism of the robot's performance -> 'user_criticism' in 'assistance'.\n"
            "- Fire, smoke, injury, medical or physical hazard -> 'safety_alert' in 'emergency'.\n"
            "- Achievements or solved tasks -> 'task_success' in 'assistance'.\n"
            "- Praise/thanks -> 'user_praise'; greetings -> 'user_greeting'.\n"
            'Return ONLY valid JSON: {"event": "<event>", "context": "<context>"}.\n\n'
            f'Text: "{text}"\nJSON:'
        )

    def _fallback_extract_event(
        self,
        text: str,
        valid_events: List[str],
        valid_contexts: List[str],
    ) -> Tuple[str, str]:
        """Weighted IT/EN event and context extractor with negation handling."""
        lower = re.sub(r"\s+", " ", text.lower().strip())
        if not lower:
            return "user_silence", "casual_conversation"

        negated_positive = bool(
            re.search(
                r"\b(non|not|never|mai)\s+(sono\s+|va\s+|sta\s+|è\s+|is\s+|feel\s+)?"
                r"(bene|felice|content[oa]|bravo|riuscito|funziona|happy|good|great|ok|well|solved)\b",
                lower,
            )
        )

        # Ordered semantic rules with weighted keyword/regex signals
        rule_specs: List[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]] = [
            (
                "safety_alert",
                "emergency",
                (
                    r"\b(fuoco|fumo|incendio|pericolo|emergenza|incidente|ospedale|sangue|"
                    r"terremoto|allagamento|sviene|svenuto|ambulan|118|112|smoke|fire|danger|"
                    r"emergency|accident|bleeding|allarme)\b",
                ),
                ("aiuto!", "help!", "soccorso"),
            ),
            (
                "user_hostility",
                "casual_conversation",
                (
                    r"\b(stupido|idiota|inutile|imbecille|deficiente|odio|schifo|vaff\w*|"
                    r"arrabbiat\w*|incazz\w*|furios[oa]|rabbia|nervos[oa]|stupid|useless|"
                    r"idiot|hate|angry|furious|shut up)\b",
                ),
                (
                    "tagliat", "spaccat", "distrutt", "rubat", "sequestrat", "tradit",
                    "urlat", "insult", "destroyed", "stole", "cut my",
                ),
            ),
            (
                "user_distress",
                "emotional_support",
                (
                    r"\b(triste|tristezza|ansia|ansioso|ansiosa|stress|stressat\w*|male|"
                    r"piang\w*|solo|sola|solitudine|stanc[oa]|preoccupat\w*|sopraffatt\w*|"
                    r"depress\w*|soffr\w*|paura|terrore|disperat\w*|lutto|mort[oa]|"
                    r"licenziat\w*|lasciat\w*|mollat\w*|litigat\w*|ferit[oa]|cacciat\w*|"
                    r"abbandonat\w*|sfrattat\w*|sad|anxious|stressed|overwhelmed|depressed|"
                    r"worried|lonely|cry|crying|kicked out|thrown out|homeless|heartbroken|grief)\b",
                ),
                (
                    "non ce la faccio", "senza casa", "buttato fuori", "messo alla porta",
                    "sta male", "sto malissimo", "mi sento vuoto", "mi manca",
                ),
            ),
            (
                "task_failure",
                "casual_conversation",
                (
                    r"\b(annullat\w*|rimandat\w*|pioggia|fallit\w*|perso|persa|rotto|rotta|"
                    r"rovinat\w*|guast\w*|errore|bug|bocciat\w*|insuffic\w*|peccato|"
                    r"multa|ritardo|sbagliat\w*|cancelled|canceled|failed|rain|broken|"
                    r"lost|ruined|bad grade|failed exam)\b",
                ),
                ("brutto voto", "non funziona", "andata male", "andato male"),
            ),
            (
                "user_criticism",
                "assistance",
                (
                    r"\b(critica|deluso|delusa|lento|lenta|pessimo|pessima|incomprensibile|"
                    r"wrong|disappointed|bad job)\b",
                ),
                ("non capisci", "hai sbagliato", "non mi aiuti", "non serve a niente"),
            ),
            (
                "user_praise",
                "casual_conversation",
                (
                    r"\b(bravo|brava|grazie|grande|fantastico|ottimo|perfetto|complimenti|"
                    r"adoro|utilissimo|thanks|thank you|awesome|great job|amazing)\b",
                ),
                (),
            ),
            (
                "task_success",
                "assistance",
                (
                    r"\b(riuscito|riuscita|risolto|vinto|funziona|promosso|promossa|"
                    r"laureat\w*|felice|content[oa]|solved|success|worked|won|happy)\b",
                ),
                ("ce l'ho fatta", "ho preso 30", "passato l'esame", "tutto bene"),
            ),
            (
                "user_greeting",
                "casual_conversation",
                (
                    r"\b(ciao|buongiorno|buonasera|buonanotte|salve|ehi|hello|hi|hey|"
                    r"good morning|good evening)\b",
                ),
                ("come stai", "how are you"),
            ),
        ]

        best_event: Optional[str] = None
        best_context: Optional[str] = None
        best_score = 0

        for event, context, regexes, substrings in rule_specs:
            if negated_positive and event in {"user_praise", "task_success"}:
                continue

            score = 0
            for pattern in regexes:
                score += len(re.findall(pattern, lower)) * 2
            for sub in substrings:
                if sub in lower:
                    score += 2

            # Slight priority boost for safety/distress over generic greetings when mixed
            if score > 0 and event in {"safety_alert", "user_distress"}:
                score += 1

            if score > best_score:
                best_score = score
                best_event = event
                best_context = context

        if best_event is None and negated_positive:
            best_event = "user_distress"
            best_context = "emotional_support"

        if best_event is not None:
            ev = best_event if best_event in valid_events else valid_events[0]
            ctx = best_context if best_context in valid_contexts else valid_contexts[0]
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

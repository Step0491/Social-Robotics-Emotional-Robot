"""Local LLM integration via Ollama grounded in the Knowledge Graph output."""

from typing import Optional
import requests

from src.knowledge_graph import ReasoningResult


class OllamaClient:
    """Client for generating context-aware robot verbal responses via local Ollama."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: float = 15.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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
            "(1 to 3 sentences) strictly grounded in the Knowledge Graph decision below. "
            "Do not contradict the selected behavior or invent new actions.\n\n"
            f"{utterance_line}"
            f"Interaction context: {reasoning.context}\n"
            f"Perceived event: {reasoning.event}\n"
            f"Robot internal emotion: {reasoning.new_emotion} (intensity: {reasoning.intensity.value})\n"
            f"Selected behavior: {reasoning.selected_behavior.name}\n"
            f"Required tone: {reasoning.selected_behavior.tone}\n"
            f"Reasoning trace:\n{trace_block}\n\n"
            "Robot response:"
        )

    def _fallback_response(self, reasoning: ReasoningResult) -> str:
        """Deterministic template fallback when Ollama is unavailable."""
        behavior_templates = {
            "warm_greeting": "Hello! It is great to see you. How can I help you today?",
            "enthusiastic_acknowledgment": "Thank you! I am really glad that went well.",
            "empathetic_listening": "I hear you, and I am here with you. Take your time if you want to share more.",
            "calm_reassurance": "Everything is under control. Let us take this one step at a time.",
            "polite_apology": "I am sorry about that. Let me adjust and help you fix this right away.",
            "proactive_check_in": "I am here whenever you are ready. Would you like assistance with anything?",
            "urgent_safety_guidance": "Attention: please pause and follow safety procedures immediately.",
        }
        return behavior_templates.get(
            reasoning.selected_behavior.name,
            "I am here to assist you.",
        )

    def generate_response(
        self,
        reasoning: ReasoningResult,
        user_utterance: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Generate a verbal response via Ollama, returning (text, used_ollama)."""
        prompt = self._build_prompt(reasoning, user_utterance)
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            text = str(data.get("response", "")).strip()
            if text:
                return text, True
        except (requests.RequestException, ValueError):
            pass

        return self._fallback_response(reasoning), False

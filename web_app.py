"""Local web server for the 3D emotional robot simulation."""

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from src.llm_client import OllamaClient
from src.rdf_exporter import export_to_turtle
from src.robot_agent import InteractionTurnResult, RobotEmotionalAgent

WEB_DIR = Path(__file__).resolve().parent / "web"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
ENV_PATH = Path(__file__).resolve().parent / ".env"


def create_default_agent() -> RobotEmotionalAgent:
    """Instantiate the default agent after loading local .env if present."""
    _load_dotenv()
    llm_client = OllamaClient(
        mode="auto",
        online_provider="groq",
        online_model="openai/gpt-oss-20b",
    )
    return RobotEmotionalAgent(llm_client=llm_client)


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load key=value pairs from .env into os.environ if not already set."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _serialize_state(agent: RobotEmotionalAgent) -> Dict[str, Any]:
    emotion = agent.state.emotion
    return {
        "emotion": emotion,
        "intensity": agent.state.intensity.value,
        "visuals": agent.ontology.get_emotion_visuals(emotion),
        "history": [
            {"event": ev, "context": ctx, "emotion": em}
            for ev, ctx, em in agent.state.history[-10:]
        ],
        "episode_count": len(agent.state.history),
        "events": agent.ontology.list_events(),
        "contexts": agent.ontology.list_contexts(),
        "graph_stats": agent.knowledge_graph.stats(),
    }


def _serialize_turn(
    agent: RobotEmotionalAgent, turn: InteractionTurnResult
) -> Dict[str, Any]:
    r = turn.reasoning
    ev_spec = agent.ontology.validate_event(r.event)
    ctx_spec = agent.ontology.validate_context(r.context)
    em_spec = agent.ontology.validate_emotion(r.new_emotion)
    visuals = agent.ontology.get_emotion_visuals(r.new_emotion)

    kg_triples = [
        f"({r.event}) --[triggers]--> ({r.new_emotion})",
        f"({r.new_emotion}) --[suggests]--> ({r.selected_behavior.name})",
        f"({r.selected_behavior.name}) --[suitable_for]--> ({r.context})",
    ]
    for ex_beh in r.excluded_behaviors:
        kg_triples.append(f"({ex_beh}) --[contraindicated_in]--> ({r.context})")

    return {
        "event": r.event,
        "context": r.context,
        "previous_emotion": r.previous_emotion,
        "emotion": r.new_emotion,
        "intensity": r.intensity.value,
        "visuals": visuals,
        "behavior": {
            "name": r.selected_behavior.name,
            "tone": r.selected_behavior.tone,
            "gesture": r.selected_behavior.gesture,
            "description": r.selected_behavior.description,
        },
        "candidate_scores": r.candidate_scores,
        "excluded_behaviors": r.excluded_behaviors,
        "reasoning_trace": r.reasoning_trace,
        "kg_triples": kg_triples,
        "ontology_validation": {
            "event_category": ev_spec.category,
            "event_valence": ev_spec.impact_valence.value,
            "context_priority": ctx_spec.priority,
            "emotion_valence": em_spec.valence.value,
            "emotion_opposite": em_spec.opposite,
        },
        "timings": {
            "extraction_ms": turn.extraction_ms,
            "kg_latency_ms": turn.kg_latency_ms,
            "llm_latency_ms": turn.llm_latency_ms,
            "total_ms": round(
                turn.extraction_ms + turn.kg_latency_ms + turn.llm_latency_ms, 3
            ),
        },
        "history": [
            {"event": ev, "context": ctx, "emotion": em}
            for ev, ctx, em in agent.state.history[-10:]
        ],
        "episode_count": len(agent.state.history),
        "verbal_response": turn.verbal_response,
        "llm_source": turn.llm_source,
        "llm_error": agent.llm_client.last_error,
        "extraction_source": turn.extraction_source,
        "graph_stats": agent.knowledge_graph.stats(),
    }


def _apply_llm_config(agent: RobotEmotionalAgent, cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        return
    if "mode" in cfg and cfg["mode"]:
        agent.llm_client.mode = str(cfg["mode"])
    if "offline_model" in cfg and cfg["offline_model"]:
        agent.llm_client.model = str(cfg["offline_model"]).strip()
    if "ollama_url" in cfg and cfg["ollama_url"]:
        agent.llm_client.base_url = str(cfg["ollama_url"]).strip().rstrip("/")
    if "online_provider" in cfg and cfg["online_provider"]:
        agent.llm_client.online_provider = str(cfg["online_provider"]).strip()
    if "online_model" in cfg:
        agent.llm_client.online_model = str(cfg["online_model"]).strip()
    if "api_key" in cfg:
        key = str(cfg["api_key"]).strip()
        agent.llm_client.api_key = key or None


def make_handler(agent: RobotEmotionalAgent):
    class RobotHTTPHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def _send_ttl(self) -> None:
            ttl = export_to_turtle(agent.knowledge_graph, agent.state).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/turtle; charset=utf-8")
            self.send_header(
                "Content-Disposition", 'attachment; filename="robot_ontology.ttl"'
            )
            self.send_header("Content-Length", str(len(ttl)))
            self.end_headers()
            self.wfile.write(ttl)

        def do_HEAD(self) -> None:  # noqa: N802
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                html_path = WEB_DIR / "index.html"
                content = html_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if self.path == "/avatar.glb":
                glb_path = (
                    PUBLIC_DIR / "avatar.glb"
                    if (PUBLIC_DIR / "avatar.glb").is_file()
                    else WEB_DIR / "avatar.glb"
                )
                if not glb_path.is_file():
                    self._send_json({"error": "avatar.glb not found"}, status=HTTPStatus.NOT_FOUND)
                    return
                content = glb_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Cache-Control", "public, max-age=3600")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

            if self.path == "/api/state":
                self._send_json(_serialize_state(agent))
                return

            if self.path == "/api/export-ttl":
                self._send_ttl()
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = self._read_json()
            except ValueError:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return

            if self.path == "/api/reset":
                agent.reset_state()
                self._send_json(_serialize_state(agent))
                return

            if self.path == "/api/export-ttl":
                if isinstance(payload.get("history"), list):
                    agent.restore_history(payload["history"])
                self._send_ttl()
                return

            if self.path == "/api/interact":
                _apply_llm_config(agent, payload.get("llm", {}))
                if isinstance(payload.get("history"), list) and len(payload["history"]) != len(
                    agent.state.history
                ):
                    agent.restore_history(payload["history"])

                text = str(payload.get("text", "")).strip()
                event = str(payload.get("event", "")).strip()
                context = str(payload.get("context", "auto")).strip()

                try:
                    if event:
                        resolved_ctx = (
                            context if context and context != "auto" else "casual_conversation"
                        )
                        turn = agent.process_interaction(
                            event=event,
                            context=resolved_ctx,
                            user_utterance=text or None,
                        )
                    elif text:
                        turn = agent.process_free_text(text=text, context=context)
                    else:
                        self._send_json(
                            {"error": "Provide either 'text' or 'event'."},
                            status=HTTPStatus.BAD_REQUEST,
                        )
                        return
                except ValueError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                    return

                self._send_json(_serialize_turn(agent, turn))
                return

            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return RobotHTTPHandler


def main() -> None:
    default_port = int(os.environ.get("PORT", "8000"))
    default_host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    parser = argparse.ArgumentParser(description="3D Emotional Robot Web Simulator")
    parser.add_argument("--host", default=default_host, help="Host address")
    parser.add_argument("--port", type=int, default=default_port, help="Port number")
    args = parser.parse_args()

    agent = create_default_agent()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent))
    print(f"Robot 3D simulator running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

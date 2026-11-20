"""Unit tests for ontology, knowledge graph reasoning, and Ollama integration."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.knowledge_graph import RobotInternalState, RobotKnowledgeGraph
from src.llm_client import OllamaClient
from src.ontology import IntensityLevel, RobotOntology
from src.rdf_exporter import export_to_turtle
from src.robot_agent import RobotEmotionalAgent


class TestRobotOntology(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = RobotOntology()

    def test_valid_entities(self) -> None:
        self.assertEqual(self.ontology.validate_emotion("joy").name, "joy")
        self.assertEqual(self.ontology.validate_event("user_greeting").name, "user_greeting")
        self.assertEqual(self.ontology.validate_context("emergency").name, "emergency")
        self.assertEqual(self.ontology.validate_behavior("calm_reassurance").name, "calm_reassurance")

    def test_unknown_entity_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.ontology.validate_event("unknown_event")
        with self.assertRaises(ValueError):
            self.ontology.validate_context("unknown_context")

    def test_behavior_constraints(self) -> None:
        self.assertFalse(self.ontology.is_behavior_allowed("warm_greeting", "emergency"))
        self.assertFalse(
            self.ontology.is_behavior_allowed("enthusiastic_acknowledgment", "emotional_support")
        )
        self.assertTrue(
            self.ontology.is_behavior_allowed("empathetic_listening", "emotional_support")
        )


class TestRobotKnowledgeGraph(unittest.TestCase):
    def setUp(self) -> None:
        self.kg = RobotKnowledgeGraph()
        self.state = RobotInternalState()

    def test_graph_populated(self) -> None:
        stats = self.kg.stats()
        self.assertGreater(stats["nodes"], 20)
        self.assertGreater(stats["edges"], 30)

    def test_greeting_triggers_joy_and_warm_greeting(self) -> None:
        result = self.kg.reason("user_greeting", "casual_conversation", self.state)
        self.assertEqual(result.new_emotion, "joy")
        self.assertEqual(result.selected_behavior.name, "warm_greeting")
        self.assertTrue(len(result.reasoning_trace) >= 2)

    def test_distress_in_emotional_support_selects_empathy(self) -> None:
        result = self.kg.reason("user_distress", "emotional_support", self.state)
        self.assertEqual(result.new_emotion, "empathy")
        self.assertEqual(result.selected_behavior.name, "empathetic_listening")

    def test_emergency_filters_restricted_behaviors_and_sets_high_intensity(self) -> None:
        result = self.kg.reason("safety_alert", "emergency", self.state)
        self.assertEqual(result.new_emotion, "fear")
        self.assertEqual(result.intensity, IntensityLevel.HIGH)
        self.assertEqual(result.selected_behavior.name, "urgent_safety_guidance")

    def test_repeated_emotion_escalates_intensity(self) -> None:
        self.kg.reason("user_praise", "casual_conversation", self.state)
        self.assertEqual(self.state.intensity, IntensityLevel.MEDIUM)
        self.kg.reason("task_success", "assistance", self.state)
        self.assertEqual(self.state.emotion, "joy")
        self.assertEqual(self.state.intensity, IntensityLevel.HIGH)


class TestOllamaClientAndAgent(unittest.TestCase):
    @patch("src.llm_client.requests.post")
    def test_ollama_success(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello! How can I assist you today?"}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        agent = RobotEmotionalAgent(llm_client=OllamaClient(model="llama3.2"))
        turn = agent.process_interaction("user_greeting", "casual_conversation", "Hi!")
        self.assertTrue(turn.used_ollama)
        self.assertIn("Hello", turn.verbal_response)

    @patch("src.llm_client.requests.post")
    def test_ollama_offline_uses_fallback(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.ConnectionError("Ollama unreachable")

        agent = RobotEmotionalAgent()
        turn = agent.process_interaction("user_distress", "emotional_support", "I feel stressed.")
        self.assertFalse(turn.used_ollama)
        self.assertEqual(turn.reasoning.selected_behavior.name, "empathetic_listening")
        self.assertTrue(len(turn.verbal_response) > 0)

    def test_reset_state(self) -> None:
        agent = RobotEmotionalAgent()
        agent.knowledge_graph.reason("user_praise", "casual_conversation", agent.state)
        self.assertEqual(agent.state.emotion, "joy")
        agent.reset_state()
        self.assertEqual(agent.state.emotion, "calm")
        self.assertEqual(len(agent.state.history), 0)

    def test_hostility_triggers_anger_with_red_visuals(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_interaction("user_hostility", "casual_conversation", "Sei inutile!")
        self.assertEqual(turn.reasoning.new_emotion, "anger")
        self.assertEqual(turn.reasoning.selected_behavior.name, "firm_boundary")
        visuals = agent.ontology.get_emotion_visuals("anger")
        self.assertEqual(visuals["expression"], "angry")
        self.assertEqual(visuals["bg_primary"], "#dc2626")

    def test_free_text_italian_event_extraction_fallback(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_free_text("partita annullata per pioggia")
        self.assertEqual(turn.reasoning.event, "task_failure")
        self.assertEqual(turn.reasoning.new_emotion, "sadness")
        visuals = agent.ontology.get_emotion_visuals("sadness")
        self.assertEqual(visuals["expression"], "sad")
        self.assertEqual(visuals["bg_primary"], "#38bdf8")

    def test_free_text_cut_cables_triggers_anger_not_greeting(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_free_text("mia madre mi ha tagliato i cavi della play station 5")
        self.assertEqual(turn.reasoning.event, "user_hostility")
        self.assertEqual(turn.reasoning.new_emotion, "anger")

    def test_free_text_unknown_does_not_default_to_greeting(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_free_text("oggi ho comprato una sedia")
        self.assertNotEqual(turn.reasoning.event, "user_greeting")

    def test_free_text_bad_grade_triggers_sadness(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_free_text("ho preso un brutto voto")
        self.assertEqual(turn.reasoning.event, "task_failure")
        self.assertEqual(turn.reasoning.new_emotion, "sadness")

    @patch("src.llm_client.requests.post")
    def test_online_gemini_provider(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": "Mi dispiace per la pioggia!"}]}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = OllamaClient(mode="online", online_provider="gemini", api_key="test-key")
        agent = RobotEmotionalAgent(llm_client=client)
        turn = agent.process_interaction(
            "task_failure", "casual_conversation", "partita annullata per pioggia"
        )
        self.assertTrue(turn.used_ollama)
        self.assertEqual(turn.llm_source, "online:gemini")
        self.assertIn("pioggia", turn.verbal_response)

    @patch("src.llm_client.requests.post")
    def test_online_groq_provider(self, mock_post: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Capisco la tua frustrazione!"}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        client = OllamaClient(mode="online", online_provider="groq", api_key="gsk-test")
        agent = RobotEmotionalAgent(llm_client=client)
        turn = agent.process_interaction("user_hostility", "casual_conversation", "Che nervoso!")
        self.assertTrue(turn.used_ollama)
        self.assertEqual(turn.llm_source, "online:groq")
        self.assertIn("frustrazione", turn.verbal_response)

    def test_free_text_kicked_out_of_home_triggers_empathy_in_italian(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        for phrase in ("mamma mi ha cacciato di casa", "mamma mi ha cacciato via di casa"):
            agent.reset_state()
            turn = agent.process_free_text(phrase)
            self.assertEqual(turn.reasoning.event, "user_distress")
            self.assertEqual(turn.reasoning.context, "emotional_support")
            self.assertEqual(turn.reasoning.new_emotion, "empathy")
            self.assertEqual(turn.reasoning.selected_behavior.name, "empathetic_listening")
            self.assertIn("Capisco", turn.verbal_response)

    def test_free_text_negated_positive_triggers_distress(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        turn = agent.process_free_text("non sono felice oggi")
        self.assertEqual(turn.reasoning.event, "user_distress")
        self.assertEqual(turn.reasoning.new_emotion, "empathy")

    def test_praise_in_casual_conversation_selects_acknowledgment(self) -> None:
        kg = RobotKnowledgeGraph()
        res = kg.reason("user_praise", "casual_conversation", RobotInternalState())
        self.assertEqual(res.new_emotion, "joy")
        self.assertEqual(res.selected_behavior.name, "enthusiastic_acknowledgment")

    @patch("src.llm_client.requests.post")
    def test_groq_http_error_records_last_error(self, mock_post: MagicMock) -> None:
        err_resp = MagicMock()
        err_resp.status_code = 401
        err_resp.json.return_value = {"error": {"message": "Invalid API Key"}}
        http_err = requests.HTTPError("401 Client Error", response=err_resp)
        mock_post.return_value.raise_for_status.side_effect = http_err

        client = OllamaClient(mode="online", online_provider="groq", api_key="gsk-bad")
        agent = RobotEmotionalAgent(llm_client=client)
        turn = agent.process_interaction("user_greeting", "casual_conversation", "Ciao")
        self.assertFalse(turn.used_ollama)
        self.assertEqual(client.last_error, "HTTP 401: Invalid API Key")

    @patch("src.llm_client.requests.get")
    @patch("src.llm_client.requests.post")
    def test_groq_deprecated_model_auto_switches(
        self, mock_post: MagicMock, mock_get: MagicMock
    ) -> None:
        dep_resp = MagicMock()
        dep_resp.status_code = 400
        dep_resp.json.return_value = {
            "error": {"message": "The model `old-groq-model` has been decommissioned."}
        }
        dep_err = requests.HTTPError("400 Bad Request", response=dep_resp)

        ok_post = MagicMock()
        ok_post.raise_for_status.return_value = None
        ok_post.json.return_value = {
            "choices": [{"message": {"content": "Ciao! Sono qui per te."}}]
        }

        fail_post = MagicMock()
        fail_post.raise_for_status.side_effect = dep_err
        mock_post.side_effect = [fail_post, ok_post]

        models_resp = MagicMock()
        models_resp.raise_for_status.return_value = None
        models_resp.json.return_value = {
            "data": [
                {"id": "whisper-large-v3", "active": True},
                {"id": "llama-3.3-70b-versatile", "active": True},
            ]
        }
        mock_get.return_value = models_resp

        client = OllamaClient(
            mode="online",
            online_provider="groq",
            online_model="old-groq-model",
            api_key="gsk-test",
        )
        agent = RobotEmotionalAgent(llm_client=client)
        turn = agent.process_interaction("user_greeting", "casual_conversation", "Ciao")
        self.assertTrue(turn.used_ollama)
        self.assertEqual(turn.llm_source, "online:groq")
        self.assertEqual(client.online_model, "llama-3.3-70b-versatile")
        self.assertEqual(client.resolved_model, "llama-3.3-70b-versatile")
        self.assertIn("switched to 'llama-3.3-70b-versatile'", client.last_model_notice or "")


class TestRDFExporter(unittest.TestCase):
    def test_turtle_export_contains_tbox_and_episodic_abox(self) -> None:
        agent = RobotEmotionalAgent()
        agent.knowledge_graph.reason("user_greeting", "casual_conversation", agent.state)
        agent.knowledge_graph.reason("user_distress", "emotional_support", agent.state)

        ttl = export_to_turtle(agent.knowledge_graph, agent.state)
        self.assertIn("robo:EmotionalOntology a owl:Ontology", ttl)
        self.assertIn("robo:Emotion a owl:Class .", ttl)
        self.assertIn("robo:user_greeting robo:triggers robo:joy .", ttl)
        self.assertIn("robo:warm_greeting robo:contraindicatedIn robo:emergency .", ttl)
        self.assertIn("robo:currentEmotion robo:empathy ;", ttl)
        self.assertIn("robo:episode_1 a robo:InteractionEpisode, owl:NamedIndividual ;", ttl)
        self.assertIn("robo:episode_2 a robo:InteractionEpisode, owl:NamedIndividual ;", ttl)

    def test_restore_history_rebuilds_episodic_state(self) -> None:
        agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
        agent.restore_history(
            [
                {"event": "user_praise", "context": "casual_conversation", "emotion": "joy"},
            ]
        )
        self.assertEqual(agent.state.emotion, "joy")
        self.assertEqual(agent.state.intensity, IntensityLevel.MEDIUM)
        turn = agent.process_interaction("task_success", "assistance")
        self.assertEqual(turn.reasoning.intensity, IntensityLevel.HIGH)
        self.assertEqual(len(agent.state.history), 2)


if __name__ == "__main__":
    unittest.main()

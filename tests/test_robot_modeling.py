"""Unit tests for ontology, knowledge graph reasoning, and Ollama integration."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.knowledge_graph import RobotInternalState, RobotKnowledgeGraph
from src.llm_client import OllamaClient
from src.ontology import IntensityLevel, RobotOntology
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


if __name__ == "__main__":
    unittest.main()

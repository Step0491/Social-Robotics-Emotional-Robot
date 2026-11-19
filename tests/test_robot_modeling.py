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


if __name__ == "__main__":
    unittest.main()

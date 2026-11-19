"""End-to-end social robot agent combining ontology, KG reasoning, and Ollama."""

from dataclasses import dataclass
from typing import Optional

from src.knowledge_graph import ReasoningResult, RobotInternalState, RobotKnowledgeGraph
from src.llm_client import OllamaClient
from src.ontology import RobotOntology


@dataclass
class InteractionTurnResult:
    """Complete output of one social interaction turn."""

    reasoning: ReasoningResult
    verbal_response: str
    used_ollama: bool


class RobotEmotionalAgent:
    """Social robot agent driven by ontology-based emotional modeling."""

    def __init__(
        self,
        ontology: Optional[RobotOntology] = None,
        knowledge_graph: Optional[RobotKnowledgeGraph] = None,
        llm_client: Optional[OllamaClient] = None,
    ) -> None:
        self.ontology = ontology or RobotOntology()
        self.knowledge_graph = knowledge_graph or RobotKnowledgeGraph(self.ontology)
        self.llm_client = llm_client or OllamaClient()
        self.state = RobotInternalState()

    def process_interaction(
        self,
        event: str,
        context: str,
        user_utterance: Optional[str] = None,
    ) -> InteractionTurnResult:
        """Process a perceived event and context to update emotion and respond."""
        reasoning = self.knowledge_graph.reason(event, context, self.state)
        verbal_response, used_ollama = self.llm_client.generate_response(
            reasoning, user_utterance=user_utterance
        )
        return InteractionTurnResult(
            reasoning=reasoning,
            verbal_response=verbal_response,
            used_ollama=used_ollama,
        )

    def reset_state(self) -> None:
        """Reset the robot's internal emotional state to baseline."""
        self.state = RobotInternalState()

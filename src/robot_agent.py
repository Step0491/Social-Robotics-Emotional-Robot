"""End-to-end social robot agent combining ontology, KG reasoning, and LLMs."""

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
    llm_source: str = "fallback"
    extraction_source: str = "direct"


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
        extraction_source: str = "direct",
    ) -> InteractionTurnResult:
        """Process a perceived event and context to update emotion and respond."""
        reasoning = self.knowledge_graph.reason(event, context, self.state)
        verbal_response, used_llm = self.llm_client.generate_response(
            reasoning, user_utterance=user_utterance
        )
        return InteractionTurnResult(
            reasoning=reasoning,
            verbal_response=verbal_response,
            used_ollama=used_llm,
            llm_source=self.llm_client.last_source,
            extraction_source=extraction_source,
        )

    def process_free_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> InteractionTurnResult:
        """Extract event/context from free text and run graph reasoning."""
        preferred_ctx = context if context and context != "auto" else None
        event, resolved_ctx, extraction_source = self.llm_client.extract_event(
            text=text,
            valid_events=self.ontology.list_events(),
            valid_contexts=self.ontology.list_contexts(),
            preferred_context=preferred_ctx,
        )
        return self.process_interaction(
            event=event,
            context=resolved_ctx,
            user_utterance=text,
            extraction_source=extraction_source,
        )

    def reset_state(self) -> None:
        """Reset the robot's internal emotional state to baseline."""
        self.state = RobotInternalState()

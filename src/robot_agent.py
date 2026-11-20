"""End-to-end social robot agent combining ontology, KG reasoning, and LLMs."""

from dataclasses import dataclass
from time import perf_counter
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
    extraction_ms: float = 0.0
    kg_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0


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
        extraction_ms: float = 0.0,
    ) -> InteractionTurnResult:
        """Process a perceived event and context to update emotion and respond."""
        t0 = perf_counter()
        reasoning = self.knowledge_graph.reason(event, context, self.state)
        kg_latency_ms = round((perf_counter() - t0) * 1000.0, 3)

        t1 = perf_counter()
        verbal_response, used_llm = self.llm_client.generate_response(
            reasoning, user_utterance=user_utterance
        )
        llm_latency_ms = round((perf_counter() - t1) * 1000.0, 3)

        return InteractionTurnResult(
            reasoning=reasoning,
            verbal_response=verbal_response,
            used_ollama=used_llm,
            llm_source=self.llm_client.last_source,
            extraction_source=extraction_source,
            extraction_ms=round(extraction_ms, 3),
            kg_latency_ms=kg_latency_ms,
            llm_latency_ms=llm_latency_ms,
        )

    def process_free_text(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> InteractionTurnResult:
        """Extract event/context from free text and run graph reasoning."""
        preferred_ctx = context if context and context != "auto" else None
        t0 = perf_counter()
        event, resolved_ctx, extraction_source = self.llm_client.extract_event(
            text=text,
            valid_events=self.ontology.list_events(),
            valid_contexts=self.ontology.list_contexts(),
            preferred_context=preferred_ctx,
        )
        extraction_ms = (perf_counter() - t0) * 1000.0
        return self.process_interaction(
            event=event,
            context=resolved_ctx,
            user_utterance=text,
            extraction_source=extraction_source,
            extraction_ms=extraction_ms,
        )

    def reset_state(self) -> None:
        """Reset the robot's internal emotional state to baseline."""
        self.state = RobotInternalState()

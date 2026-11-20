"""Knowledge Graph representation and explainable reasoning for robot emotions."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from src.ontology import (
    EntityType,
    IntensityLevel,
    RelationType,
    RobotBehaviorSpec,
    RobotOntology,
    Valence,
)


@dataclass
class RobotInternalState:
    """Tracks the robot's current emotional state and interaction history."""

    emotion: str = "calm"
    intensity: IntensityLevel = IntensityLevel.LOW
    history: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class ReasoningResult:
    """Explainable output produced by Knowledge Graph traversal."""

    event: str
    context: str
    previous_emotion: str
    new_emotion: str
    intensity: IntensityLevel
    selected_behavior: RobotBehaviorSpec
    candidate_scores: Dict[str, float]
    excluded_behaviors: Dict[str, str]
    reasoning_trace: List[str]


class RobotKnowledgeGraph:
    """Knowledge Graph linking events, contexts, robot emotions, and behaviors."""

    def __init__(self, ontology: Optional[RobotOntology] = None) -> None:
        self.ontology = ontology or RobotOntology()
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._build_graph()

    def _add_node(self, name: str, entity_type: EntityType) -> None:
        self.graph.add_node(name, entity_type=entity_type.value)

    def _add_edge(
        self,
        source: str,
        target: str,
        relation: RelationType,
        weight: float = 1.0,
    ) -> None:
        self.graph.add_edge(source, target, relation=relation.value, weight=weight)

    def _build_graph(self) -> None:
        """Populate nodes and typed relationships from the ontology and domain rules."""
        for emotion_name, spec in self.ontology.emotions.items():
            self._add_node(emotion_name, EntityType.EMOTION)
            if spec.opposite:
                self._add_edge(emotion_name, spec.opposite, RelationType.HAS_OPPOSITE)

        for event_name in self.ontology.events:
            self._add_node(event_name, EntityType.EVENT)

        for context_name in self.ontology.contexts:
            self._add_node(context_name, EntityType.CONTEXT)

        for behavior_name, spec in self.ontology.behaviors.items():
            self._add_node(behavior_name, EntityType.BEHAVIOR)
            for restricted_ctx in spec.restricted_contexts:
                self._add_edge(
                    behavior_name, restricted_ctx, RelationType.CONTRAINDICATED_IN
                )

        # Event -> Emotion transitions
        event_emotion_links = [
            ("user_greeting", "joy", 0.9),
            ("user_greeting", "trust", 0.7),
            ("user_praise", "joy", 1.0),
            ("user_praise", "trust", 0.8),
            ("user_distress", "empathy", 1.0),
            ("user_distress", "calm", 0.6),
            ("user_criticism", "sadness", 0.8),
            ("user_criticism", "calm", 0.5),
            ("user_silence", "anticipation", 0.8),
            ("user_silence", "calm", 0.7),
            ("task_success", "joy", 0.9),
            ("task_success", "anticipation", 0.6),
            ("task_failure", "sadness", 0.9),
            ("task_failure", "empathy", 0.5),
            ("safety_alert", "fear", 1.0),
            ("safety_alert", "surprise", 0.7),
            ("user_hostility", "anger", 1.0),
            ("user_hostility", "calm", 0.4),
        ]
        for event_name, emotion_name, weight in event_emotion_links:
            self._add_edge(event_name, emotion_name, RelationType.TRIGGERS, weight=weight)

        # Emotion -> Behavior mappings
        emotion_behavior_links = [
            ("joy", "warm_greeting", 0.9),
            ("joy", "enthusiastic_acknowledgment", 1.0),
            ("trust", "warm_greeting", 0.8),
            ("trust", "proactive_check_in", 0.8),
            ("empathy", "empathetic_listening", 1.0),
            ("empathy", "calm_reassurance", 0.9),
            ("sadness", "polite_apology", 1.0),
            ("sadness", "calm_reassurance", 0.7),
            ("anticipation", "proactive_check_in", 0.9),
            ("anticipation", "warm_greeting", 0.5),
            ("calm", "calm_reassurance", 0.8),
            ("calm", "empathetic_listening", 0.7),
            ("calm", "proactive_check_in", 0.6),
            ("fear", "urgent_safety_guidance", 1.0),
            ("fear", "calm_reassurance", 0.6),
            ("surprise", "urgent_safety_guidance", 0.8),
            ("surprise", "proactive_check_in", 0.7),
            ("anger", "firm_boundary", 1.0),
            ("anger", "calm_reassurance", 0.75),
        ]
        for emotion_name, behavior_name, weight in emotion_behavior_links:
            self._add_edge(
                emotion_name, behavior_name, RelationType.SUGGESTS, weight=weight
            )

        # Behavior -> Context suitability boosts
        suitable_links = [
            ("warm_greeting", "casual_conversation", 0.4),
            ("enthusiastic_acknowledgment", "casual_conversation", 0.3),
            ("enthusiastic_acknowledgment", "assistance", 0.3),
            ("empathetic_listening", "emotional_support", 0.5),
            ("calm_reassurance", "emotional_support", 0.4),
            ("calm_reassurance", "emergency", 0.3),
            ("polite_apology", "assistance", 0.4),
            ("polite_apology", "emotional_support", 0.2),
            ("proactive_check_in", "assistance", 0.4),
            ("proactive_check_in", "casual_conversation", 0.2),
            ("urgent_safety_guidance", "emergency", 0.6),
            ("firm_boundary", "casual_conversation", 0.4),
            ("firm_boundary", "assistance", 0.3),
        ]
        for behavior_name, context_name, weight in suitable_links:
            self._add_edge(
                behavior_name, context_name, RelationType.SUITABLE_FOR, weight=weight
            )

    def _compute_intensity(
        self,
        new_emotion: str,
        state: RobotInternalState,
        context_name: str,
    ) -> IntensityLevel:
        """Adjust emotional intensity based on context priority and state continuity."""
        emotion_spec = self.ontology.validate_emotion(new_emotion)
        context_spec = self.ontology.validate_context(context_name)

        if context_spec.priority >= 5:
            return IntensityLevel.HIGH

        if state.emotion == new_emotion:
            if state.intensity == IntensityLevel.LOW:
                return IntensityLevel.MEDIUM
            return IntensityLevel.HIGH

        return emotion_spec.default_intensity

    def _select_emotion(
        self,
        event_name: str,
        context_name: str,
        state: RobotInternalState,
    ) -> Tuple[str, IntensityLevel, List[str]]:
        """Determine the next emotional state via TRIGGERS edges."""
        trace: List[str] = []
        triggered: List[Tuple[str, float]] = []

        for _, target, data in self.graph.out_edges(event_name, data=True):
            if data.get("relation") == RelationType.TRIGGERS.value:
                triggered.append((target, float(data.get("weight", 1.0))))

        if not triggered:
            return state.emotion, state.intensity, [
                f"No direct emotion trigger found for '{event_name}'; retaining '{state.emotion}'."
            ]

        triggered.sort(key=lambda item: item[1], reverse=True)
        selected_emotion, top_weight = triggered[0]

        # In emotional_support context, prioritize empathy over generic calm on negative events
        event_spec = self.ontology.validate_event(event_name)
        if (
            context_name == "emotional_support"
            and event_spec.impact_valence == Valence.NEGATIVE
            and any(em == "empathy" for em, _ in triggered)
        ):
            selected_emotion = "empathy"
            trace.append(
                f"Context '{context_name}' prioritized 'empathy' for negative event '{event_name}'."
            )
        else:
            trace.append(
                f"Event '{event_name}' triggered emotion '{selected_emotion}' (weight={top_weight:.2f})."
            )

        intensity = self._compute_intensity(selected_emotion, state, context_name)
        trace.append(
            f"Robot state updated: '{state.emotion}' -> '{selected_emotion}' (intensity={intensity.value})."
        )
        return selected_emotion, intensity, trace

    def reason(
        self,
        event_name: str,
        context_name: str,
        state: RobotInternalState,
    ) -> ReasoningResult:
        """Run graph-based reasoning to update emotion and select a feasible behavior."""
        self.ontology.validate_event(event_name)
        self.ontology.validate_context(context_name)

        previous_emotion = state.emotion
        new_emotion, intensity, trace = self._select_emotion(
            event_name, context_name, state
        )

        # Retrieve candidate behaviors via Emotion -> Behavior edges
        candidates: Dict[str, float] = {}
        excluded: Dict[str, str] = {}

        for _, behavior_name, data in self.graph.out_edges(new_emotion, data=True):
            if data.get("relation") != RelationType.SUGGESTS.value:
                continue

            base_weight = float(data.get("weight", 1.0))

            # Check feasibility constraints (CONTRAINDICATED_IN)
            if not self.ontology.is_behavior_allowed(behavior_name, context_name):
                reason_msg = (
                    f"Contraindicated in context '{context_name}' by ontology constraints."
                )
                excluded[behavior_name] = reason_msg
                trace.append(f"Excluded behavior '{behavior_name}': {reason_msg}")
                continue

            # Boost score if explicitly suitable for the active context
            context_boost = 0.0
            for _, ctx_target, ctx_data in self.graph.out_edges(behavior_name, data=True):
                if (
                    ctx_data.get("relation") == RelationType.SUITABLE_FOR.value
                    and ctx_target == context_name
                ):
                    context_boost += float(ctx_data.get("weight", 0.0))

            candidates[behavior_name] = round(base_weight + context_boost, 3)

        # Safe fallback if all emotion-linked behaviors were filtered out
        if not candidates:
            fallback_name = (
                "urgent_safety_guidance"
                if context_name == "emergency"
                else "calm_reassurance"
            )
            candidates[fallback_name] = 0.5
            trace.append(
                f"All primary candidates excluded; applied safe fallback '{fallback_name}'."
            )

        best_behavior_name = max(candidates, key=lambda k: candidates[k])
        selected_behavior = self.ontology.validate_behavior(best_behavior_name)
        trace.append(
            f"Selected behavior '{best_behavior_name}' (score={candidates[best_behavior_name]:.2f}, "
            f"tone='{selected_behavior.tone}', gesture='{selected_behavior.gesture}')."
        )

        # Mutate state after successful reasoning
        state.emotion = new_emotion
        state.intensity = intensity
        state.history.append((event_name, context_name, new_emotion))

        return ReasoningResult(
            event=event_name,
            context=context_name,
            previous_emotion=previous_emotion,
            new_emotion=new_emotion,
            intensity=intensity,
            selected_behavior=selected_behavior,
            candidate_scores=candidates,
            excluded_behaviors=excluded,
            reasoning_trace=trace,
        )

    def stats(self) -> Dict[str, int]:
        """Return node and edge counts of the Knowledge Graph."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

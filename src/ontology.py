"""Formal ontology definitions for robot emotional states and social behaviors."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, List, Optional


class EntityType(str, Enum):
    """Node types in the robot emotional ontology."""

    EMOTION = "Emotion"
    EVENT = "PerceivedEvent"
    CONTEXT = "InteractionContext"
    BEHAVIOR = "RobotBehavior"


class RelationType(str, Enum):
    """Typed relationships connecting ontology entities."""

    TRIGGERS = "triggers"
    SUGGESTS = "suggests"
    SUITABLE_FOR = "suitable_for"
    CONTRAINDICATED_IN = "contraindicated_in"
    HAS_OPPOSITE = "has_opposite"


class IntensityLevel(str, Enum):
    """Discrete emotional intensity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Valence(str, Enum):
    """Emotional valence polarity."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class EmotionalStateSpec:
    """Definition of a robot internal emotional state."""

    name: str
    valence: Valence
    default_intensity: IntensityLevel
    description: str
    opposite: Optional[str] = None


@dataclass(frozen=True)
class PerceivedEventSpec:
    """Definition of an external social or task event perceived by the robot."""

    name: str
    category: str
    impact_valence: Valence
    description: str


@dataclass(frozen=True)
class InteractionContextSpec:
    """Definition of the active social interaction context."""

    name: str
    priority: int
    description: str


@dataclass(frozen=True)
class RobotBehaviorSpec:
    """Definition of a candidate behavioral response of the robot."""

    name: str
    tone: str
    gesture: str
    description: str
    restricted_contexts: FrozenSet[str] = field(default_factory=frozenset)


class RobotOntology:
    """Controlled vocabulary and structural constraints for robot emotional modeling."""

    def __init__(self) -> None:
        self.emotions: Dict[str, EmotionalStateSpec] = {
            "joy": EmotionalStateSpec(
                name="joy",
                valence=Valence.POSITIVE,
                default_intensity=IntensityLevel.MEDIUM,
                description="Positive affective state triggered by social warmth or success.",
                opposite="sadness",
            ),
            "trust": EmotionalStateSpec(
                name="trust",
                valence=Valence.POSITIVE,
                default_intensity=IntensityLevel.MEDIUM,
                description="Cooperative and open state fostering rapport with the user.",
                opposite="fear",
            ),
            "empathy": EmotionalStateSpec(
                name="empathy",
                valence=Valence.NEUTRAL,
                default_intensity=IntensityLevel.MEDIUM,
                description="Attentive, supportive internal state in response to user distress.",
                opposite="anger",
            ),
            "surprise": EmotionalStateSpec(
                name="surprise",
                valence=Valence.NEUTRAL,
                default_intensity=IntensityLevel.MEDIUM,
                description="Heightened alertness caused by unexpected input or obstacles.",
                opposite="anticipation",
            ),
            "anticipation": EmotionalStateSpec(
                name="anticipation",
                valence=Valence.POSITIVE,
                default_intensity=IntensityLevel.LOW,
                description="Proactive readiness for upcoming user requests or tasks.",
                opposite="surprise",
            ),
            "sadness": EmotionalStateSpec(
                name="sadness",
                valence=Valence.NEGATIVE,
                default_intensity=IntensityLevel.MEDIUM,
                description="Low-arousal negative state following failure or user criticism.",
                opposite="joy",
            ),
            "fear": EmotionalStateSpec(
                name="fear",
                valence=Valence.NEGATIVE,
                default_intensity=IntensityLevel.HIGH,
                description="Cautious state triggered by safety risks or emergency signals.",
                opposite="trust",
            ),
            "calm": EmotionalStateSpec(
                name="calm",
                valence=Valence.NEUTRAL,
                default_intensity=IntensityLevel.LOW,
                description="Baseline balanced internal state.",
                opposite=None,
            ),
            "anger": EmotionalStateSpec(
                name="anger",
                valence=Valence.NEGATIVE,
                default_intensity=IntensityLevel.HIGH,
                description="Defensive high-arousal state triggered by hostility or disrespect.",
                opposite="empathy",
            ),
        }

        self.emotion_visuals: Dict[str, Dict[str, str]] = {
            "joy": {
                "bg_primary": "#f59e0b",
                "bg_secondary": "#5c3406",
                "accent": "#fde047",
                "expression": "happy",
            },
            "trust": {
                "bg_primary": "#10b981",
                "bg_secondary": "#064e3b",
                "accent": "#6ee7b7",
                "expression": "warm",
            },
            "empathy": {
                "bg_primary": "#8b5cf6",
                "bg_secondary": "#2e1065",
                "accent": "#c4b5fd",
                "expression": "empathetic",
            },
            "surprise": {
                "bg_primary": "#06b6d4",
                "bg_secondary": "#164e63",
                "accent": "#67e8f9",
                "expression": "surprised",
            },
            "anticipation": {
                "bg_primary": "#f97316",
                "bg_secondary": "#431407",
                "accent": "#fdba74",
                "expression": "curious",
            },
            "sadness": {
                "bg_primary": "#38bdf8",
                "bg_secondary": "#0c2d48",
                "accent": "#93c5fd",
                "expression": "sad",
            },
            "fear": {
                "bg_primary": "#ef4444",
                "bg_secondary": "#450a0a",
                "accent": "#fca5a5",
                "expression": "fearful",
            },
            "anger": {
                "bg_primary": "#dc2626",
                "bg_secondary": "#5a0b0b",
                "accent": "#f87171",
                "expression": "angry",
            },
            "calm": {
                "bg_primary": "#14b8a6",
                "bg_secondary": "#0f2937",
                "accent": "#5eead4",
                "expression": "neutral",
            },
        }

        self.events: Dict[str, PerceivedEventSpec] = {
            "user_greeting": PerceivedEventSpec(
                name="user_greeting",
                category="social",
                impact_valence=Valence.POSITIVE,
                description="User initiates interaction with a greeting.",
            ),
            "user_praise": PerceivedEventSpec(
                name="user_praise",
                category="social",
                impact_valence=Valence.POSITIVE,
                description="User expresses appreciation or positive feedback.",
            ),
            "user_distress": PerceivedEventSpec(
                name="user_distress",
                category="affective",
                impact_valence=Valence.NEGATIVE,
                description="User shows signs of anxiety, sadness, or stress.",
            ),
            "user_criticism": PerceivedEventSpec(
                name="user_criticism",
                category="social",
                impact_valence=Valence.NEGATIVE,
                description="User expresses dissatisfaction with the robot.",
            ),
            "user_silence": PerceivedEventSpec(
                name="user_silence",
                category="interaction",
                impact_valence=Valence.NEUTRAL,
                description="Extended pause with no user input.",
            ),
            "task_success": PerceivedEventSpec(
                name="task_success",
                category="task",
                impact_valence=Valence.POSITIVE,
                description="Assigned assistance or action completed successfully.",
            ),
            "task_failure": PerceivedEventSpec(
                name="task_failure",
                category="task",
                impact_valence=Valence.NEGATIVE,
                description="Assigned assistance or action failed to complete.",
            ),
            "safety_alert": PerceivedEventSpec(
                name="safety_alert",
                category="safety",
                impact_valence=Valence.NEGATIVE,
                description="Potential hazard or urgent situation detected.",
            ),
            "user_hostility": PerceivedEventSpec(
                name="user_hostility",
                category="social",
                impact_valence=Valence.NEGATIVE,
                description="User uses insults, aggression, or hostile language.",
            ),
        }

        self.contexts: Dict[str, InteractionContextSpec] = {
            "casual_conversation": InteractionContextSpec(
                name="casual_conversation",
                priority=1,
                description="Informal everyday dialogue with the user.",
            ),
            "assistance": InteractionContextSpec(
                name="assistance",
                priority=2,
                description="Goal-oriented support for a user task.",
            ),
            "emotional_support": InteractionContextSpec(
                name="emotional_support",
                priority=3,
                description="Sensitive interaction focused on user well-being.",
            ),
            "emergency": InteractionContextSpec(
                name="emergency",
                priority=5,
                description="High-priority safety or urgent intervention context.",
            ),
        }

        self.behaviors: Dict[str, RobotBehaviorSpec] = {
            "warm_greeting": RobotBehaviorSpec(
                name="warm_greeting",
                tone="friendly and welcoming",
                gesture="open_wave",
                description="Greet the user warmly and invite interaction.",
                restricted_contexts=frozenset({"emergency"}),
            ),
            "enthusiastic_acknowledgment": RobotBehaviorSpec(
                name="enthusiastic_acknowledgment",
                tone="warm and appreciative",
                gesture="happy_nod",
                description="Express gratitude and positive engagement.",
                restricted_contexts=frozenset({"emergency", "emotional_support"}),
            ),
            "empathetic_listening": RobotBehaviorSpec(
                name="empathetic_listening",
                tone="gentle, supportive, and attentive",
                gesture="attentive_lean",
                description="Validate user feelings and offer calm presence.",
                restricted_contexts=frozenset(),
            ),
            "calm_reassurance": RobotBehaviorSpec(
                name="calm_reassurance",
                tone="steady, calm, and reassuring",
                gesture="open_palms",
                description="Provide composed reassurance and reduce tension.",
                restricted_contexts=frozenset(),
            ),
            "polite_apology": RobotBehaviorSpec(
                name="polite_apology",
                tone="humble and constructive",
                gesture="slight_bow",
                description="Acknowledge shortcomings and offer corrective action.",
                restricted_contexts=frozenset(),
            ),
            "proactive_check_in": RobotBehaviorSpec(
                name="proactive_check_in",
                tone="curious and polite",
                gesture="gentle_head_tilt",
                description="Gently ask if the user needs assistance.",
                restricted_contexts=frozenset({"emergency"}),
            ),
            "urgent_safety_guidance": RobotBehaviorSpec(
                name="urgent_safety_guidance",
                tone="clear, direct, and calm",
                gesture="alert_posture",
                description="Deliver concise safety instructions immediately.",
                restricted_contexts=frozenset({"casual_conversation"}),
            ),
            "firm_boundary": RobotBehaviorSpec(
                name="firm_boundary",
                tone="firm, respectful, and composed",
                gesture="raised_hand",
                description="Set a respectful boundary while de-escalating hostility.",
                restricted_contexts=frozenset({"emergency"}),
            ),
        }

    def validate_emotion(self, emotion: str) -> EmotionalStateSpec:
        """Return EmotionalStateSpec or raise ValueError if unknown."""
        if emotion not in self.emotions:
            raise ValueError(f"Unknown emotion in ontology: '{emotion}'")
        return self.emotions[emotion]

    def validate_event(self, event: str) -> PerceivedEventSpec:
        """Return PerceivedEventSpec or raise ValueError if unknown."""
        if event not in self.events:
            raise ValueError(f"Unknown perceived event in ontology: '{event}'")
        return self.events[event]

    def validate_context(self, context: str) -> InteractionContextSpec:
        """Return InteractionContextSpec or raise ValueError if unknown."""
        if context not in self.contexts:
            raise ValueError(f"Unknown interaction context in ontology: '{context}'")
        return self.contexts[context]

    def validate_behavior(self, behavior: str) -> RobotBehaviorSpec:
        """Return RobotBehaviorSpec or raise ValueError if unknown."""
        if behavior not in self.behaviors:
            raise ValueError(f"Unknown robot behavior in ontology: '{behavior}'")
        return self.behaviors[behavior]

    def is_behavior_allowed(self, behavior: str, context: str) -> bool:
        """Check whether a behavior is feasible in the given context."""
        spec = self.validate_behavior(behavior)
        self.validate_context(context)
        return context not in spec.restricted_contexts

    def list_events(self) -> List[str]:
        """Return all valid perceived event names."""
        return sorted(self.events.keys())

    def list_contexts(self) -> List[str]:
        """Return all valid interaction context names."""
        return sorted(self.contexts.keys())

    def get_emotion_visuals(self, emotion: str) -> Dict[str, str]:
        """Return visual color and expression profile for an emotion."""
        self.validate_emotion(emotion)
        return dict(self.emotion_visuals[emotion])

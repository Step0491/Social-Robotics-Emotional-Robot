"""W3C Turtle (OWL/RDF) exporter for the robot emotional ontology and episodic ABox."""

from typing import List, Optional

from src.knowledge_graph import RobotInternalState, RobotKnowledgeGraph
from src.ontology import RelationType


RELATION_PREDICATES = {
    RelationType.TRIGGERS.value: "robo:triggers",
    RelationType.SUGGESTS.value: "robo:suggests",
    RelationType.SUITABLE_FOR.value: "robo:suitableFor",
    RelationType.CONTRAINDICATED_IN.value: "robo:contraindicatedIn",
    RelationType.HAS_OPPOSITE.value: "robo:hasOpposite",
}


def _escape_literal(text: str) -> str:
    """Escape characters for valid Turtle string literals."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def export_to_turtle(
    kg: RobotKnowledgeGraph,
    state: Optional[RobotInternalState] = None,
) -> str:
    """Serialize TBox ontology, Knowledge Graph edges, and episodic ABox to Turtle."""
    lines: List[str] = [
        "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
        "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
        "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
        "@prefix robo: <http://uniba.it/social-robotics/emotional-ontology#> .",
        "",
        "robo:EmotionalOntology a owl:Ontology ;",
        '    rdfs:label "Ontology-Based Robot Emotional Modeling"@en .',
        "",
        "# --- TBox: Classes ---",
    ]

    for cls_name in (
        "Emotion",
        "PerceivedEvent",
        "InteractionContext",
        "RobotBehavior",
        "InteractionEpisode",
    ):
        lines.append(f"robo:{cls_name} a owl:Class .")

    lines.extend(["", "# --- TBox: Properties ---"])
    for prop_name in (
        "triggers",
        "suggests",
        "suitableFor",
        "contraindicatedIn",
        "hasOpposite",
        "currentEmotion",
        "perceivedEvent",
        "inContext",
        "resultingEmotion",
    ):
        lines.append(f"robo:{prop_name} a owl:ObjectProperty .")

    for dprop_name in ("hasValence", "hasIntensity", "hasTone", "hasGesture", "stepIndex"):
        lines.append(f"robo:{dprop_name} a owl:DatatypeProperty .")

    lines.extend(["", "# --- Domain Individuals: Emotions ---"])
    for name, spec in kg.ontology.emotions.items():
        lines.append(
            f"robo:{name} a robo:Emotion, owl:NamedIndividual ;\n"
            f'    robo:hasValence "{_escape_literal(spec.valence.value)}"^^xsd:string ;\n'
            f'    robo:hasIntensity "{_escape_literal(spec.default_intensity.value)}"^^xsd:string ;\n'
            f'    rdfs:comment "{_escape_literal(spec.description)}"@en .'
        )

    lines.extend(["", "# --- Domain Individuals: Perceived Events ---"])
    for name, spec in kg.ontology.events.items():
        lines.append(
            f"robo:{name} a robo:PerceivedEvent, owl:NamedIndividual ;\n"
            f'    robo:hasValence "{_escape_literal(spec.impact_valence.value)}"^^xsd:string ;\n'
            f'    rdfs:comment "{_escape_literal(spec.description)}"@en .'
        )

    lines.extend(["", "# --- Domain Individuals: Interaction Contexts ---"])
    for name, spec in kg.ontology.contexts.items():
        lines.append(
            f"robo:{name} a robo:InteractionContext, owl:NamedIndividual ;\n"
            f'    rdfs:comment "{_escape_literal(spec.description)}"@en .'
        )

    lines.extend(["", "# --- Domain Individuals: Robot Behaviors ---"])
    for name, spec in kg.ontology.behaviors.items():
        lines.append(
            f"robo:{name} a robo:RobotBehavior, owl:NamedIndividual ;\n"
            f'    robo:hasTone "{_escape_literal(spec.tone)}"^^xsd:string ;\n'
            f'    robo:hasGesture "{_escape_literal(spec.gesture)}"^^xsd:string ;\n'
            f'    rdfs:comment "{_escape_literal(spec.description)}"@en .'
        )

    lines.extend(["", "# --- Knowledge Graph Relations ---"])
    for source, target, data in kg.graph.edges(data=True):
        relation = data.get("relation")
        predicate = RELATION_PREDICATES.get(relation)
        if predicate:
            lines.append(f"robo:{source} {predicate} robo:{target} .")

    if state is not None:
        lines.extend(
            [
                "",
                "# --- ABox: Dynamic Robot State & Episodic Memory ---",
                "robo:RobotCurrentState a owl:NamedIndividual ;",
                f"    robo:currentEmotion robo:{state.emotion} ;",
                f'    robo:hasIntensity "{_escape_literal(state.intensity.value)}"^^xsd:string .',
            ]
        )
        for idx, (ev, ctx, em) in enumerate(state.history, start=1):
            lines.append(
                f"robo:episode_{idx} a robo:InteractionEpisode, owl:NamedIndividual ;\n"
                f'    robo:stepIndex "{idx}"^^xsd:integer ;\n'
                f"    robo:perceivedEvent robo:{ev} ;\n"
                f"    robo:inContext robo:{ctx} ;\n"
                f"    robo:resultingEmotion robo:{em} ."
            )

    return "\n".join(lines) + "\n"

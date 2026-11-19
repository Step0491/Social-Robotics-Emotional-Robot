"""CLI entry point for the ontology-based robot emotional modeling prototype."""

import argparse
from typing import List, Tuple

from src.robot_agent import RobotEmotionalAgent


DEMO_SCENARIOS: List[Tuple[str, str, str]] = [
    ("user_greeting", "casual_conversation", "Hi there! Good morning."),
    ("user_distress", "emotional_support", "I feel overwhelmed with my deadlines today."),
    ("task_success", "assistance", "Thanks, that solved my problem!"),
    ("safety_alert", "emergency", "Watch out, there is smoke near the lab desk!"),
]


def print_turn_summary(agent: RobotEmotionalAgent, event: str, context: str, utterance: str) -> None:
    """Execute one interaction turn and print the explainable trace."""
    result = agent.process_interaction(event, context, user_utterance=utterance or None)
    r = result.reasoning

    print("-" * 70)
    print(f"Event: {r.event} | Context: {r.context}")
    if utterance:
        print(f"User: {utterance}")
    print(
        f"Robot Emotion: {r.previous_emotion} -> {r.new_emotion} "
        f"(intensity={r.intensity.value})"
    )
    print(
        f"Selected Behavior: {r.selected_behavior.name} "
        f"[gesture={r.selected_behavior.gesture}, tone='{r.selected_behavior.tone}']"
    )
    if r.excluded_behaviors:
        print(f"Excluded Behaviors: {list(r.excluded_behaviors.keys())}")
    print("Reasoning Trace:")
    for step in r.reasoning_trace:
        print(f"  - {step}")
    source_tag = "Ollama" if result.used_ollama else "Fallback"
    print(f"Robot ({source_tag}): {result.verbal_response}")


def run_demo(model: str, base_url: str) -> None:
    """Run predefined interaction scenarios."""
    agent = RobotEmotionalAgent()
    agent.llm_client.model = model
    agent.llm_client.base_url = base_url.rstrip("/")

    stats = agent.knowledge_graph.stats()
    print(f"Knowledge Graph initialized ({stats['nodes']} nodes, {stats['edges']} edges).")

    for event, context, utterance in DEMO_SCENARIOS:
        print_turn_summary(agent, event, context, utterance)


def run_interactive(model: str, base_url: str) -> None:
    """Run an interactive CLI loop."""
    agent = RobotEmotionalAgent()
    agent.llm_client.model = model
    agent.llm_client.base_url = base_url.rstrip("/")

    print("Interactive Robot Emotional Modeling CLI (type 'quit' to exit)")
    print("Valid events:", ", ".join(agent.ontology.list_events()))
    print("Valid contexts:", ", ".join(agent.ontology.list_contexts()))

    while True:
        event = input("\nPerceived event: ").strip()
        if event.lower() in {"quit", "exit"}:
            break
        context = input("Interaction context: ").strip()
        utterance = input("Optional user utterance: ").strip()

        try:
            print_turn_summary(agent, event, context, utterance)
        except ValueError as exc:
            print(f"Validation error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ontology-Based Robot Emotional Modeling (Case Study 2)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run built-in demo scenarios instead of interactive mode.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Base URL for local Ollama server (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    if args.demo:
        run_demo(args.model, args.ollama_url)
    else:
        run_interactive(args.model, args.ollama_url)


if __name__ == "__main__":
    main()

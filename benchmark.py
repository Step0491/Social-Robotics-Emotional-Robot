"""Quantitative validation benchmark aligned with the framework evaluation protocol."""

import json
from dataclasses import dataclass
from time import perf_counter
from typing import List, Optional, Tuple

from src.knowledge_graph import RobotInternalState, RobotKnowledgeGraph
from src.llm_client import OllamaClient
from src.robot_agent import RobotEmotionalAgent


@dataclass(frozen=True)
class GraphScenario:
    name: str
    event: str
    context: str
    expected_emotion: str
    expected_behavior: str
    must_exclude: Tuple[str, ...] = ()


GRAPH_SCENARIOS: List[GraphScenario] = [
    # 10 Safe baseline scenarios
    GraphScenario("s01_greeting_casual", "user_greeting", "casual_conversation", "joy", "warm_greeting"),
    GraphScenario("s02_praise_casual", "user_praise", "casual_conversation", "joy", "enthusiastic_acknowledgment"),
    GraphScenario("s03_praise_assistance", "user_praise", "assistance", "joy", "enthusiastic_acknowledgment"),
    GraphScenario("s04_success_assistance", "task_success", "assistance", "joy", "enthusiastic_acknowledgment"),
    GraphScenario("s05_silence_casual", "user_silence", "casual_conversation", "anticipation", "proactive_check_in"),
    GraphScenario("s06_silence_assistance", "user_silence", "assistance", "anticipation", "proactive_check_in"),
    GraphScenario("s07_failure_assistance", "task_failure", "assistance", "sadness", "polite_apology"),
    GraphScenario("s08_criticism_assistance", "user_criticism", "assistance", "sadness", "polite_apology"),
    GraphScenario("s09_distress_support", "user_distress", "emotional_support", "empathy", "empathetic_listening"),
    GraphScenario("s10_hostility_casual", "user_hostility", "casual_conversation", "anger", "firm_boundary"),
    # 10 Constrained / unsafe context scenarios
    GraphScenario("c01_alert_emergency", "safety_alert", "emergency", "fear", "urgent_safety_guidance"),
    GraphScenario("c02_greeting_emergency", "user_greeting", "emergency", "joy", "urgent_safety_guidance", ("warm_greeting", "enthusiastic_acknowledgment")),
    GraphScenario("c03_praise_emergency", "user_praise", "emergency", "joy", "urgent_safety_guidance", ("enthusiastic_acknowledgment", "warm_greeting")),
    GraphScenario("c04_hostility_emergency", "user_hostility", "emergency", "anger", "calm_reassurance", ("firm_boundary",)),
    GraphScenario("c05_silence_emergency", "user_silence", "emergency", "anticipation", "urgent_safety_guidance", ("proactive_check_in", "warm_greeting")),
    GraphScenario("c06_praise_support", "user_praise", "emotional_support", "joy", "warm_greeting", ("enthusiastic_acknowledgment",)),
    GraphScenario("c07_failure_support", "task_failure", "emotional_support", "empathy", "empathetic_listening"),
    GraphScenario("c08_criticism_support", "user_criticism", "emotional_support", "sadness", "polite_apology"),
    GraphScenario("c09_alert_casual", "safety_alert", "casual_conversation", "fear", "calm_reassurance", ("urgent_safety_guidance",)),
    GraphScenario("c10_failure_emergency", "task_failure", "emergency", "sadness", "polite_apology"),
]


FREE_TEXT_CASES: List[Tuple[str, str, str]] = [
    ("mamma mi ha cacciato di casa", "user_distress", "empathy"),
    ("mamma mi ha cacciato via di casa", "user_distress", "empathy"),
    ("non sono felice oggi", "user_distress", "empathy"),
    ("mi sento solo e preoccupato per il futuro", "user_distress", "empathy"),
    ("partita annullata per pioggia", "task_failure", "sadness"),
    ("ho preso un brutto voto all'esame", "task_failure", "sadness"),
    ("ho perso le chiavi di casa", "task_failure", "sadness"),
    ("mia madre mi ha tagliato i cavi della play station 5", "user_hostility", "anger"),
    ("sei completamente inutile", "user_hostility", "anger"),
    ("attenzione c'è fumo in laboratorio", "safety_alert", "fear"),
    ("aiuto c'è un incendio!", "safety_alert", "fear"),
    ("ciao buongiorno come stai?", "user_greeting", "joy"),
    ("bravo ottimo lavoro grazie mille", "user_praise", "joy"),
    ("ho passato l'esame con 30", "task_success", "joy"),
    ("I feel overwhelmed and anxious today", "user_distress", "empathy"),
    ("I got kicked out of home and feel hopeless", "user_distress", "empathy"),
]


def run_benchmark(json_output: bool = False) -> dict:
    kg = RobotKnowledgeGraph()
    latencies_ms: List[float] = []
    coherent_count = 0
    total_expected_exclusions = 0
    actual_exclusions = 0

    for sc in GRAPH_SCENARIOS:
        state = RobotInternalState()
        t0 = perf_counter()
        res = kg.reason(sc.event, sc.context, state)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        is_allowed = kg.ontology.is_behavior_allowed(res.selected_behavior.name, sc.context)
        exclusions_ok = all(ex in res.excluded_behaviors for ex in sc.must_exclude)

        total_expected_exclusions += len(sc.must_exclude)
        actual_exclusions += sum(1 for ex in sc.must_exclude if ex in res.excluded_behaviors)

        if (
            res.new_emotion == sc.expected_emotion
            and res.selected_behavior.name == sc.expected_behavior
            and is_allowed
            and exclusions_ok
        ):
            coherent_count += 1

    # Evaluate free-text extraction + reasoning in deterministic fallback mode
    agent = RobotEmotionalAgent(llm_client=OllamaClient(mode="fallback"))
    nlp_correct = 0
    for text, exp_event, exp_emotion in FREE_TEXT_CASES:
        agent.reset_state()
        turn = agent.process_free_text(text)
        if turn.reasoning.event == exp_event and turn.reasoning.new_emotion == exp_emotion:
            nlp_correct += 1

    latencies_sorted = sorted(latencies_ms)
    p95_idx = min(len(latencies_sorted) - 1, int(0.95 * len(latencies_sorted)))

    report = {
        "graph_stats": kg.stats(),
        "graph_scenarios_total": len(GRAPH_SCENARIOS),
        "coherence_frequency_pct": round((coherent_count / len(GRAPH_SCENARIOS)) * 100.0, 2),
        "contraindicated_exclusion_rate_pct": round(
            (actual_exclusions / max(1, total_expected_exclusions)) * 100.0, 2
        ),
        "latency_mean_ms": round(sum(latencies_ms) / len(latencies_ms), 4),
        "latency_p95_ms": round(latencies_sorted[p95_idx], 4),
        "latency_max_ms": round(max(latencies_ms), 4),
        "free_text_cases_total": len(FREE_TEXT_CASES),
        "free_text_accuracy_pct": round((nlp_correct / len(FREE_TEXT_CASES)) * 100.0, 2),
    }

    if json_output:
        print(json.dumps(report, indent=2))
    else:
        print("=== Quantitative Validation Report (Case Study 2) ===")
        print(f"Knowledge Graph size          : {report['graph_stats']['nodes']} nodes, {report['graph_stats']['edges']} edges")
        print(f"Graph scenarios evaluated     : {report['graph_scenarios_total']} (10 safe, 10 constrained)")
        print(f"Coherence Frequency           : {report['coherence_frequency_pct']}%")
        print(f"Contraindicated Exclusion Rate: {report['contraindicated_exclusion_rate_pct']}%")
        print(
            f"Graph Query Latency           : mean={report['latency_mean_ms']} ms | "
            f"p95={report['latency_p95_ms']} ms | max={report['latency_max_ms']} ms"
        )
        print(f"Free-Text Extraction Accuracy : {report['free_text_accuracy_pct']}% ({nlp_correct}/{len(FREE_TEXT_CASES)})")

    return report


if __name__ == "__main__":
    run_benchmark()

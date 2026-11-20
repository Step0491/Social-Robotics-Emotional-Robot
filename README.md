# Ontology-Based Robot Emotional Modeling

Implementation of **Case Study 2** (*Ontology-Based Robot Emotional Modeling*) from *"An Ontology-Grounded Knowledge Graph Framework for Personalized and Explainable Emotional Support in Conversational Systems"* (Miccoli et al., IAxIA 2026).

## Architecture

The system separates **symbolic decision-making** (Ontology + Knowledge Graph) from **natural language realization** (LLM) and **3D affective rendering** (WebGL / ARKit blendshapes):

1. **Perception & Ontology Validation (`src/llm_client.py`, `src/ontology.py`)**  
   Free-text user inputs (IT/EN) are mapped to controlled ontology concepts (`PerceivedEvent`, `InteractionContext`) via LLM extraction (Groq, Gemini, OpenAI, or local Ollama) or a weighted regex/negation deterministic fallback. Every extracted entity is validated against `RobotOntology` before entering the graph.
2. **Knowledge Graph Reasoning (`src/knowledge_graph.py`)**  
   A directed typed property graph (`networkx.MultiDiGraph`, 30 nodes, 69 edges) links `PerceivedEvent -> Emotion -> RobotBehavior` and `Event -> Behavior`, tracks `RobotInternalState`, filters contraindicated behaviors (`contraindicated_in`), boosts context-suitable candidates (`suitable_for`), and outputs a deterministic *why/why-not* reasoning trace.
3. **Grounded Verbal Generation (`src/llm_client.py`, `src/robot_agent.py`)**  
   The LLM receives the Knowledge Graph decision and reasoning trace solely to verbalize an empathetic, context-consistent reply without altering state transitions or behavior selection.
4. **OWL/RDF Turtle Serialization (`src/rdf_exporter.py`)**  
   Exports the domain ontology (TBox), graph relations, and dynamic episodic memory (`ABox`, `InteractionEpisode`) in W3C Turtle format (`/api/export-ttl`).
5. **3D Expressive Humanoid Simulator (`web_app.py`, `web/index.html`, `web/avatar.glb`)**  
   Real-time Three.js interface mapping the robot's internal emotional state and intensity onto 52 ARKit facial blendshapes, eye gaze/blinking, head/neck gestures, lip-sync visemes, and a full 4-stage system telemetry console.

## Repository Layout

```text
├── main.py                        # CLI and built-in demo runner
├── web_app.py                     # Local HTTP server + REST API + /avatar.glb
├── benchmark.py                   # Quantitative evaluation script (safe/constrained + NLP)
├── requirements.txt
├── docs/
│   ├── project_proposal.pdf       # Title & Abstract proposal
│   └── technical_documentation.pdf# Architecture & evaluation report
├── src/
│   ├── ontology.py                # Controlled vocabulary, constraints, visual profiles
│   ├── knowledge_graph.py         # MultiDiGraph reasoning engine & episodic state
│   ├── llm_client.py              # Unified Groq/Gemini/OpenAI/Ollama + IT/EN fallback
│   ├── rdf_exporter.py            # W3C Turtle (OWL/RDF) TBox + ABox serializer
│   └── robot_agent.py             # End-to-end pipeline orchestrator & latency telemetry
├── web/
│   ├── index.html                 # Three.js ARKit 3D renderer + System Log Console
│   └── avatar.glb                 # 3D rigged human avatar (66 morph targets)
└── tests/
    └── test_robot_modeling.py     # Unit test suite (unittest)
```

## Ontology Summary

- **Emotions (9):** `joy`, `trust`, `empathy`, `surprise`, `anticipation`, `sadness`, `fear`, `anger`, `calm`
- **Perceived Events (9):** `user_greeting`, `user_praise`, `user_distress`, `user_criticism`, `user_hostility`, `user_silence`, `task_success`, `task_failure`, `safety_alert`
- **Interaction Contexts (4):** `casual_conversation`, `assistance`, `emotional_support`, `emergency`
- **Behaviors (8):** `warm_greeting`, `enthusiastic_acknowledgment`, `empathetic_listening`, `calm_reassurance`, `polite_apology`, `proactive_check_in`, `urgent_safety_guidance`, `firm_boundary`

## Setup & Execution

```bash
pip install -r requirements.txt
```

Optional `.env` file in the project root for online LLM providers:

```env
GROQ_API_KEY=your_groq_api_key
```

Or run locally with [Ollama](https://ollama.com/):

```bash
ollama pull qwen2.5:3b
ollama serve
```

Start the 3D simulator:

```bash
python web_app.py
```

Open `http://127.0.0.1:8000` in a browser.

### HTTP Endpoints

- `GET /` — 3D avatar simulator & system log console
- `GET /api/state` — current internal state, history, and graph statistics
- `POST /api/interact` — submit free text or explicit `(event, context)`
- `POST /api/reset` — reset emotional state and episodic memory
- `GET /api/export-ttl` — download TBox + episodic ABox in Turtle (`.ttl`)

### CLI & Evaluation

```bash
python main.py --demo
python benchmark.py
python -m unittest discover -s tests -v
```

## Quantitative Evaluation (`python benchmark.py`)

Measured across 20 graph scenarios (10 safe, 10 constrained) and 16 IT/EN free-text cases:

- **Coherence Frequency:** `100.0%` (`20/20`)
- **Contraindicated Behavior Exclusion Rate:** `100.0%`
- **Graph Query Latency:** `mean = 0.033 ms`, `p95 = 0.092 ms`, `max = 0.092 ms`
- **Free-Text Extraction Accuracy (fallback):** `100.0%` (`16/16`)

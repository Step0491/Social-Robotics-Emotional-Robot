# Ontology-Based Robot Emotional Modeling

Python prototype implementing **Case Study 2** (*Ontology-Based Robot Emotional Modeling*) from the framework *"An Ontology-Grounded Knowledge Graph Framework for Personalized and Explainable Emotional Support in Conversational Systems"*.

## Overview

The system models the internal emotional state of a social robot and selects appropriate behavioral responses to social interactions through:

1. **Ontology Design** (`src/ontology.py`): Formal definitions of robot internal emotional states, perceived events, interaction contexts, behavioral responses, and feasibility constraints.
2. **Knowledge Graph** (`src/knowledge_graph.py`): A directed typed graph (`networkx`) linking perceived events, contexts, emotional states, and robot behaviors with explainable reasoning traces.
3. **Local LLM Integration** (`src/llm_client.py`): Integration with a locally deployed LLM via **Ollama** to generate context-aware verbal responses grounded in the Knowledge Graph output (with a deterministic fallback when Ollama is offline).
4. **Robot Agent** (`src/robot_agent.py`): End-to-end pipeline connecting perception, graph reasoning, state updates, and verbal generation.

## Setup

```bash
pip install -r requirements.txt
```

Optional: start [Ollama](https://ollama.com/) locally and pull a lightweight offline model (default: `qwen2.5:3b`):

```bash
ollama pull qwen2.5:3b
ollama serve
```

Online LLM providers (Google Gemini or OpenAI-compatible APIs) are also supported via environment variables (`GEMINI_API_KEY` / `OPENAI_API_KEY`) or directly from the web UI.

## Run

Run the **3D Web Simulator** (humanoid face, dynamic emotion backgrounds, free-text events, and Knowledge Graph trace):

```bash
python web_app.py
```

Then open `http://127.0.0.1:8000` in your browser.

Or run the CLI simulation / demo scenarios:

```bash
python main.py --demo
python main.py
```

## Test

```bash
python -m unittest discover -s tests -v
```

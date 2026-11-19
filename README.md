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

Optional: start [Ollama](https://ollama.com/) locally and pull a model (default: `llama3.2`):

```bash
ollama pull llama3.2
ollama serve
```

## Run

Run the interactive simulation or demo scenarios:

```bash
python main.py --demo
python main.py
```

## Test

```bash
python -m unittest discover -s tests -v
```

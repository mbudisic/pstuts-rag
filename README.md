---
title: PsTuts RAG
emoji: 💻
colorFrom: yellow
colorTo: indigo
sdk: docker
pinned: false
license: mit
short_description: Agentic RAG that interrogates the PsTuts dataset.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🤖 PsTuts RAG System

An agentic RAG system for the PsTuts dataset that provides AI-powered answers to Adobe Photoshop questions using video tutorial transcripts.

## 🚀 Getting Started

1. Install dependencies:

```bash
# Basic installation (includes Jupyter support)
pip install -e .

# With development tools
pip install -e ".[dev]"

# With web server components
pip install -e ".[web]"

# With additional extras (numpy, ragas, tavily)
pip install -e ".[extras]"

# Full installation with all features
pip install -e ".[dev,web,extras]"
```

2. Run the app:
```bash
chainlit run app.py
```

3. Open your browser and navigate to `http://localhost:8000`

## 💡 Features

- Retrieval-augmented generation (RAG) for Photoshop tutorials
- Multi-agent system with team supervisor
- Web search integration via Tavily
- Semantic chunking for better context retrieval
- Interactive chat interface through Chainlit

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

## 🎨 Visual Theme

The application features a beautiful **sepia-toned color scheme** that gives it a vintage, artistic feel perfect for Adobe Photoshop tutorials:

- 🌅 **Light Theme** (default): Warm cream and tan colors reminiscent of old photography
- 🌙 **Dark Theme**: Rich coffee and amber tones for comfortable nighttime usage
- 🖼️ **Sepia Filter**: Subtle sepia treatment on images for visual consistency
- ⚡ **Smooth Transitions**: Elegant animations when switching between themes

Users can toggle between light and dark variants using the theme switcher in the interface.

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

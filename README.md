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

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/mbudisic/PsTuts-RAG?label=version&sort=semver)

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

🤗 HuggingFace is updated from Github.

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
- 🌐 **Web Search Permission Control**: You can now set whether the AI is allowed to perform web searches directly from the chat settings (top right ⚙️). Choose between "Ask every time", "Always allow", or "Never allow" for full control!

## ⚙️ Configuration Options

You can customize the behavior of PsTuts RAG using environment variables. Set these in your shell, `.env` file, or deployment environment. Here are the available options:

| Env Var | Description |
|---------|-------------|
| `EVA_WORKFLOW_NAME` | 🏷️ Name of the EVA workflow. Default: `EVA_workflow` |
| `EVA_LOG_LEVEL` | 🪵 Logging level for EVA. Default: `INFO` |
| `TRANSCRIPT_GLOB` | 📄 Glob pattern for transcript JSON files (supports multiple files separated by `:`). Default: `data/test.json` |
| `EMBEDDING_MODEL` | 🧊 Name of the embedding model to use (default: custom fine-tuned snowflake model). Default: `mbudisic/snowflake-arctic-embed-s-ft-pstuts` |
| `EVA_STRIP_THINK` | 💭 If set (present in env), strips 'think' steps from EVA output. |
| `EMBEDDING_API` | 🔌 API provider for embeddings (`OPENAI`, `HUGGINGFACE`, or `OLLAMA`). Default: `HUGGINGFACE` |
| `LLM_API` | 🤖 API provider for LLM (`OPENAI`, `HUGGINGFACE`, or `OLLAMA`). Default: `OLLAMA` |
| `MAX_RESEARCH_LOOPS` | 🔁 Maximum number of research loops to perform. Default: `3` |
| `LLM_TOOL_MODEL` | 🛠️ Name of the LLM model to use for tool calling. Default: `smollm2:1.7b-instruct-q2_K` |
| `N_CONTEXT_DOCS` | 📚 Number of context documents to retrieve for RAG. Default: `2` |
| `EVA_SEARCH_PERMISSION` | 🌐 Permission for search (`yes`, `no`, or `ask`). Default: `no`. **Can also be set in the chat UI!** |
| `EVA_DB_PERSIST` | 💾 Path or flag for DB persistence. Default: unset |
| `EVA_REINITIALIZE` | 🔄 If true, reinitializes EVA DB. Default: `False` |
| `THREAD_ID` | 🧵 Thread ID for the current session. Default: unset |

Set these variables to control model selection, logging, search permissions, and more. For advanced usage, see the developer documentation.

![EVA](public/logo_dark.png)

# 📼 EVA 

EVA, the Enhanced Video Archive, provides intelligent search and retrieval capabilities for video content using RAG (Retrieval-Augmented Generation), making video content easily searchable and accessible. While the [Adobe Photoshop video tutorial dataset (PsTuts)](https://huggingface.co/datasets/mbudisic/PsTuts-VQA) is used as an example, the tool is designed to work with any video transcript dataset. 📚

![QR Code](public/QR_code.png)

## ⚠️ WORK IN PROGRESS ⚠️ 

This is a work-in-progress project, developed by [👤 **Marko Budisic**](https://www.linkedin.com/in/marko-budisic/) during 
[AI Makerspace](https://aimakerspace.io/) Cohort 6. 
If you are interested in jump starting your own AI education,
I cannot think of a better place than AI Makerspace.

Here is the [video of the project demo](https://www.youtube.com/watch?v=q3e5oHQ4K5g) as presented in June 2025.

Features to be built:

1. Automatic transcription of `.mp4` (currently relying on transcripts provided with PsTuts)
2. OCR of the video to provide a second source of context.
3. Search-by-screenshot


## ⚙️ Configuration via Environment Variables

The tool's behavior is controlled through **environment variables**, giving you complete flexibility in setup:

- 🌐 **Remote AI APIs**: Use external AI services (OpenAI, HuggingFace) for processing
- 🏠 **Ollama APIs**: Run local AI models for privacy and offline operation
- 🔧 **Hybrid Mode**: Mix and match APIs based on your needs

### Key Configuration Options:
- `LLM_API`: Choose between `OPENAI`, `HUGGINGFACE`, or `OLLAMA`
- `EMBEDDING_API`: Select embedding provider (`OPENAI`, `HUGGINGFACE`, or `OLLAMA`)
- `EVA_SEARCH_PERMISSION`: Control web search permissions (`yes`, `no`, or `ask`)

## 🔑 Key Code Components

Here are **5 essential parts** of the codebase with direct links to the source:

1. **[Multi-Agent System](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/graph.py#L258)** - Orchestrates AI agents for video search and web research
2. **[RAG Implementation](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/rag.py#L231)** - Core retrieval-augmented generation logic
3. **[Configuration](https://github.com/mbudisic/pstuts-rag/blob/main/env.example)** - APIs selectable via environment variables
4. **[Context chunking](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/datastore.py#L498)** - Timestamp-enriched document chunking
5. **[Main Application](https://github.com/mbudisic/pstuts-rag/blob/main/app.py#L68)** - Chainlit web interface and user interaction




## 🛠️ Installation Instructions

### Prerequisites
- Python 3.8+
- Git
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/) (recommended)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mbudisic/pstuts-rag.git
   cd pstuts-rag
   ```

2. **Install dependencies with uv**:
   ```bash
   # Basic installation
   uv sync
   
   # With development tools
   uv sync --extra dev
   
   # With web server components
   uv sync --extra web
   
   # Full installation with all features
   uv sync --extra dev --extra web --extra extras
   ```

3. **Set up environment variables**:
   ```bash
   # Copy the example environment file
   cp env.example .env
   
   # Edit .env file with your actual API keys and configuration
   # The template includes all available options with explanations
   ```

4. **Run the application**:
   ```bash
   uv run chainlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8000`

### Advanced Configuration

For detailed configuration options, see the [Developer Documentation](docs/DEVELOPER.md) which includes:
- Complete environment variable reference
- API integration guides
- Custom model configuration
- Performance optimization tips

---

**Ready to enhance your video archive?** 🚀 Start with the installation steps above and explore the power of AI-driven video content retrieval!

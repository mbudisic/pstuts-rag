# 📼 Enhanced Video Archive Tool

An **enhanced video archive tool** that provides intelligent search and retrieval capabilities for video content using advanced RAG (Retrieval-Augmented Generation) technology. 🎯

## 🗂️ About This Tool

This is a **video archive enhancement system** designed to make video content easily searchable and accessible. While the **PsTuts dataset is used as an example**, the tool is designed to work with any video transcript dataset. 📚

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

1. **[Multi-Agent System](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/graph.py)** - Orchestrates AI agents for video search and web research
2. **[RAG Implementation](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/rag.py)** - Core retrieval-augmented generation logic
3. **[Configuration Management](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/configuration.py)** - Handles environment variables and settings
4. **[Vector Database](https://github.com/mbudisic/pstuts-rag/blob/main/pstuts_rag/pstuts_rag/datastore.py)** - Manages Qdrant vector store and document processing
5. **[Main Application](https://github.com/mbudisic/pstuts-rag/blob/main/app.py)** - Chainlit web interface and user interaction

## 👤 Author

Developed by **[Marko Budisic](https://www.linkedin.com/in/marko-budisic/)** - Connect with me on LinkedIn! 🤝

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
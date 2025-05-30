# 🛠️ Developer Documentation

## 📦 Project Structure

```
.
├── app.py                # Main Chainlit application
├── app_simple_rag.py     # Simplified RAG application 
├── pyproject.toml        # Project configuration and dependencies
├── pstuts_rag/           # Core package
│   └── pstuts_rag/       # Source code
│       ├── __init__.py
│       ├── datastore.py  # Vector database management
│       ├── loader.py     # Data loading utilities
│       ├── rag.py        # RAG implementation
│       ├── agents.py     # Team agent implementation
│       └── ...
├── data/                 # Dataset files
└── README.md             # User documentation
```

## 🧩 Dependency Structure

Dependencies are organized into logical groups:

- **Core**: Basic dependencies needed for the RAG system (includes Jupyter support)
- **Dev**: Development tools (linting, testing, etc.)
- **Web**: Dependencies for web server functionality
- **Extras**: Additional optional packages (numpy, ragas, tavily)

You can install different combinations using pip's extras syntax:
```bash
pip install -e ".[dev,web]"  # Install core + dev + web dependencies
```

## 🔧 Technical Details

The application uses LangChain, LangGraph, and Chainlit to create an agentic RAG system:

### Key Components

- **DatastoreManager**: Manages the Qdrant vector store and document retrieval
- **RAGChainFactory**: Creates retrieval-augmented generation chains
- **PsTutsTeamState**: Manages the state of the agent-based system
- **Langgraph**: Implements the routing logic between different agents

## 🚀 Running Locally

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e ".[dev]"  # Install with development tools
```

3. Set up API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional, for web search
```

4. Run the application:
```bash
chainlit run app.py
```

## 🧪 Code Quality

To check for dependency issues:
```bash
deptry .
```

For linting:
```bash
black .
ruff check .
mypy .
```

## 📚 Resources

- [Chainlit Documentation](https://docs.chainlit.io)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) 
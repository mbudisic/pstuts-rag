# 🛠️ Developer Documentation

## 📦 Project Structure

```
.
├── app.py                        # Main Chainlit application (multi-agent RAG)
├── app_simple_rag.py             # Simplified single-agent RAG application 
├── Dockerfile                    # Docker container configuration
├── pyproject.toml                # Project configuration and dependencies
├── requirements.txt              # Basic requirements (for legacy compatibility)
├── uv.lock                       # Lock file for uv package manager
├── pstuts_rag/                   # Package directory
│   ├── pstuts_rag/               # Source code
│   │   ├── __init__.py           # Package initialization
│   │   ├── configuration.py     # Application configuration settings
│   │   ├── datastore.py          # Vector database and document management
│   │   ├── rag.py                # RAG chain implementation and factories
│   │   ├── graph.py              # LangGraph multi-agent implementation
│   │   ├── state.py              # Team state management for agents
│   │   ├── prompts.py            # System prompts for different agents
│   │   ├── evaluator_utils.py    # RAG evaluation utilities
│   │   └── utils.py              # General utilities
│   ├── setup.py                  # Package setup (legacy)
│   └── CERT_SUBMISSION.md        # Certification submission documentation
├── data/                         # Dataset files (JSON format)
│   ├── train.json                # Training dataset
│   ├── dev.json                  # Development dataset
│   ├── test.json                 # Test dataset
│   ├── kg_*.json                 # Knowledge graph datasets
│   ├── LICENSE.txt               # Dataset license
│   └── README.md                 # Dataset documentation
├── notebooks/                    # Jupyter notebooks for development
│   ├── evaluate_rag.ipynb        # RAG evaluation notebook
│   ├── transcript_rag.ipynb      # Basic RAG experiments
│   ├── transcript_agents.ipynb   # Multi-agent experiments
│   ├── Fine_Tuning_Embedding_for_PSTuts.ipynb  # Embedding fine-tuning
│   └── */                        # Fine-tuned model checkpoints
├── docs/                         # Documentation
│   ├── DEVELOPER.md              # This file - developer documentation
│   ├── ANSWER.md                 # Technical answer documentation
│   ├── BLOGPOST*.md              # Blog post drafts
│   ├── dataset_card.md           # Dataset card documentation
│   ├── TODO.md                   # Development TODO list
│   └── chainlit.md               # Chainlit welcome message
├── scripts/                      # Utility scripts (currently empty)
└── README.md                     # User-facing documentation
```

## 🧩 Dependency Structure

Dependencies are organized into logical groups in `pyproject.toml`:

### Core Dependencies 🎯
All required dependencies for the RAG system including:
- **LangChain ecosystem**: `langchain`, `langchain-core`, `langchain-community`, `langchain-openai`, `langgraph`
- **Vector database**: `qdrant-client`, `langchain-qdrant`
- **ML/AI libraries**: `sentence-transformers`, `transformers`, `torch`
- **Web interface**: `chainlit==2.0.4`
- **Data processing**: `pandas`, `datasets`, `pyarrow`
- **Evaluation**: `ragas==0.2.15`
- **Jupyter support**: `ipykernel`, `jupyter`, `ipywidgets`
- **API integration**: `tavily-python` (web search), `requests`, `python-dotenv`

### Optional Dependencies 🔧
- **dev**: Development tools (`pytest`, `black`, `mypy`, `deptry`, `ipdb`)
- **web**: Web server components (`fastapi`, `uvicorn`, `python-multipart`)

Installation examples:
```bash
pip install -e .                    # Core only
pip install -e ".[dev]"            # Core + development tools
pip install -e ".[dev,web]"        # Core + dev + web server
```

## 🔧 Technical Architecture

### Key Components

#### 🏗️ Core Classes and Factories
- **`Configuration`** (`configuration.py`): Application settings including model names, file paths, and parameters
- **`DatastoreManager`** (`datastore.py`): Manages Qdrant vector store, document loading, and semantic chunking
- **`RAGChainFactory`** (`rag.py`): Creates retrieval-augmented generation chains with reference compilation
- **`RAGChainInstance`** (`rag.py`): Encapsulates complete RAG instances with embeddings and vector stores

#### 🕸️ Multi-Agent System
- **`PsTutsTeamState`** (`state.py`): TypedDict managing multi-agent conversation state
- **Agent creation functions** (`graph.py`): Factory functions for different agent types:
  - `create_rag_node()`: Video search agent using RAG
  - `create_tavily_node()`: Adobe Help web search agent
  - `create_team_supervisor()`: LLM-based routing supervisor
- **LangGraph implementation**: Multi-agent coordination with state management

#### 📊 Document Processing
- **`VideoTranscriptBulkLoader`**: Loads entire video transcripts as single documents
- **`VideoTranscriptChunkLoader`**: Loads individual transcript segments with timestamps
- **`chunk_transcripts()`**: Async semantic chunking with timestamp preservation
- **Custom embedding models**: Fine-tuned embeddings for PsTuts domain

#### ⚡ Asynchronous Loading System
- **`DatastoreManager.loading_complete`**: AsyncIO Event object that's set when data loading completes
- **`DatastoreManager.is_ready()`**: Convenience method to check if loading is complete
- **`DatastoreManager.wait_for_loading(timeout)`**: Async method to wait for loading completion with optional timeout
- **`DatastoreManager.add_completion_callback(callback)`**: Register callbacks (sync or async) to be called when loading completes
- **Non-blocking startup**: Vector database loading runs in background threads to prevent UI blocking
- **Background processing**: `asyncio.create_task()` used for concurrent data loading during application startup
- **Event-driven notifications**: Hook into loading completion for reactive programming patterns

#### 🔍 Evaluation System
- **`evaluator_utils.py`**: RAG evaluation utilities using RAGAS framework
- **Notebook-based evaluation**: `evaluate_rag.ipynb` for systematic testing

## 🚀 Running the Applications

### Multi-Agent RAG (Recommended) 🤖
```bash
chainlit run app.py
```
Features team of agents including video search and web search capabilities.

### Simple RAG (Basic) 🔍
```bash
chainlit run app_simple_rag.py
```
Single-agent RAG system for straightforward queries.

## 🔬 Development Workflow

1. **Environment Setup**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

2. **Environment Variables**:
```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # Optional, for web search
```

3. **Code Quality Tools**:
```bash
# Dependency analysis
deptry .

# Code formatting and linting
black .
ruff check .
mypy .

# Development debugging
ipdb  # Available for interactive debugging
```

4. **Notebook Development**:
   - Use `notebooks/` for experimentation
   - `evaluate_rag.ipynb` for systematic evaluation
   - Fine-tuning experiments in `Fine_Tuning_Embedding_for_PSTuts.ipynb`

## 🏗️ Architecture Notes

- **Embedding models**: Uses custom fine-tuned `snowflake-arctic-embed-s-ft-pstuts` by default
- **Vector store**: Qdrant with semantic chunking for optimal retrieval
- **LLM**: GPT-4.1-mini for generation and routing
- **Web search**: Tavily integration targeting `helpx.adobe.com`
- **State management**: LangGraph for multi-agent coordination
- **Evaluation**: RAGAS framework for retrieval and generation metrics

## 📚 Resources

- [Chainlit Documentation](https://docs.chainlit.io) 📖
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) 🦜
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) 🕸️
- [Qdrant Documentation](https://qdrant.tech/documentation/) 🔍
- [RAGAS Documentation](https://docs.ragas.io/) 📊 

### 🔄 Usage Examples

#### Event-Based Loading with Callbacks
```python
# Option 1: Custom callback passed to startup
async def my_completion_handler():
    print("✅ Database is ready for queries!")
    await notify_users("System ready")

datastore = await startup(
    config=my_config,
    on_loading_complete=my_completion_handler
)

# Option 2: Register callbacks after initialization
datastore = await startup(config=my_config)

# Add additional callbacks
def on_complete():
    print("✅ Loading finished!")

async def on_complete_async():
    await send_notification("Database ready")

datastore.add_completion_callback(on_complete)
datastore.add_completion_callback(on_complete_async)

# Option 3: Wait for completion with timeout
if await datastore.wait_for_loading(timeout=60):
    print("Loading completed within timeout")
else:
    print("Loading timed out")
``` 
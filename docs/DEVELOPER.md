# 🛠️ Developer Documentation

> **Note:** The root-level `DEVELOPER.md` is deprecated. This is the canonical developer documentation. 🚦

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
│   │   ├── configuration.py      # Application configuration settings
│   │   ├── datastore.py          # Vector database and document management
│   │   ├── rag.py                # RAG chain implementation and factories
│   │   ├── rag_for_transcripts.py# RAG chain for video transcripts (reference packing)
│   │   ├── graph.py              # Agent node creation and LangGraph assembly
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
├── public/                       # Theme and static files (see theming section)
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
- **`Datastore`** (`datastore.py`): Manages Qdrant vector store, document loading, and semantic chunking
- **`RAGChainFactory`** (`rag.py`): Creates retrieval-augmented generation chains with reference compilation
- **`RAGChainInstance`** (`rag.py`): Encapsulates complete RAG instances with embeddings and vector stores
- **`RAG for Transcripts`** (`rag_for_transcripts.py`): Implements the RAG chain for searching video transcripts, including reference packing and post-processing for AIMessage responses. Used for context-rich, reference-annotated answers from video data. 🎬
- **`Graph Assembly`** (`graph.py`): Handles agent node creation, LangGraph assembly, and integration of multi-agent workflows. Provides utilities for building, initializing, and running the agentic graph. 🕸️

#### 🗄️ QdrantClientSingleton (datastore.py)
- **Purpose:** Ensures only one instance of QdrantClient exists per process, preventing accidental concurrent access to embedded Qdrant. Thread-safe and logs every access!
- **Usage:**
  ```python
  from pstuts_rag.datastore import QdrantClientSingleton
  client = QdrantClientSingleton.get_client(path="/path/to/db")  # or path=None for in-memory
  ```
- **Behavior:**
  - First call determines the storage location (persistent or in-memory)
  - All subsequent calls return the same client, regardless of path
  - Thread-safe via a lock
  - Every call logs the requested path for debugging 🪵

#### 🏪 Datastore (datastore.py)
- **Collection Creation Logic:**
  - On initialization, always tries to create the Qdrant collection for the vector store.
  - If the collection already exists, catches the `ValueError` and simply fetches the existing collection instead (no crash, no duplicate creation!).
  - This is the recommended robust pattern for Qdrant local mode. 🦺
  - Example log output:
    ```
    Collection EVA_AI_transcripts created.
    # or
    Collection EVA_AI_transcripts already exists.
    ```

#### 🕸️ Multi-Agent System
- **`PsTutsTeamState`** (`state.py`): TypedDict managing multi-agent conversation state
- **Agent creation functions** (`graph.py`): Factory functions for different agent types:
  - `create_rag_node()`: Video search agent using RAG
  - `create_tavily_node()`: Adobe Help web search agent
  - `create_team_supervisor()`: LLM-based routing supervisor
- **LangGraph implementation**: Multi-agent coordination with state management

#### 🛠️ Interactive Interrupt System
The system includes a sophisticated interrupt mechanism that allows for human-in-the-loop decision making during workflow execution.

**Key Features:**
- **Permission-based search control**: Users can grant or deny permission for web searches on a per-query basis
- **Real-time interrupts**: Workflow pauses execution to request user input when search permission is set to "ASK" 
- **Graceful fallback**: System continues with local RAG search if web search is denied
- **State persistence**: Search permission decisions are maintained throughout the session

**Implementation Details:**
- **`YesNoAsk` enum**: Manages three permission states - `YES`, `NO`, and `ASK`
- **Interrupt points**: Built into the `search_help` node using LangGraph's `interrupt()` function
- **Configuration control**: Default permission behavior set via `EVA_SEARCH_PERMISSION` environment variable
- **Interactive prompts**: Users receive clear yes/no prompts with automatic parsing

**Usage Workflow:**
1. User submits a query requiring web search
2. If `search_permission = ASK`, system pauses with interrupt prompt
3. User responds with "yes" to permit search or any other response to deny
4. System logs the decision and continues with appropriate search strategy
5. Permission state persists for the current session

This feature enables controlled access to external resources while maintaining autonomous operation when permissions are pre-configured. 🤖✋

#### 📊 Document Processing
- **`VideoTranscriptBulkLoader`**: Loads entire video transcripts as single documents
- **`VideoTranscriptChunkLoader`**: Loads individual transcript segments with timestamps
- **`chunk_transcripts()`**: Async semantic chunking with timestamp preservation
- **Custom embedding models**: Fine-tuned embeddings for PsTuts domain

#### ⚡ Asynchronous Loading System
- **`Datastore.loading_complete`**: AsyncIO Event object that's set when data loading completes
- **`Datastore.is_ready()`**: Convenience method to check if loading is complete
- **`Datastore.wait_for_loading(timeout)`**: Async method to wait for loading completion with optional timeout
- **`Datastore.add_completion_callback(callback)`**: Register callbacks (sync or async) to be called when loading completes
- **Non-blocking startup**: Vector database loading runs in background threads to prevent UI blocking
- **Background processing**: `asyncio.create_task()` used for concurrent data loading during application startup
- **Event-driven notifications**: Hook into loading completion for reactive programming patterns

#### 🔍 Evaluation System
- **`evaluator_utils.py`**: RAG evaluation utilities using RAGAS framework
- **Notebook-based evaluation**: `evaluate_rag.ipynb` for systematic testing

### ⚙️ Configuration Reference

The `Configuration` class (in `pstuts_rag/configuration.py`) is powered by Pydantic and supports environment variable overrides for all fields. Below is a reference for all configuration options:

| Field | Env Var | Type | Default | Description |
|-------|---------|------|---------|-------------|
| `eva_workflow_name` | `EVA_WORKFLOW_NAME` | `str` | `EVA_workflow` | 🏷️ Name of the EVA workflow |
| `eva_log_level` | `EVA_LOG_LEVEL` | `str` | `INFO` | 🪵 Logging level for EVA |
| `transcript_glob` | `TRANSCRIPT_GLOB` | `str` | `data/test.json` | 📄 Glob pattern for transcript JSON files (supports `:` for multiple) |
| `embedding_model` | `EMBEDDING_MODEL` | `str` | `mbudisic/snowflake-arctic-embed-s-ft-pstuts` | 🧊 Embedding model name (default: custom fine-tuned snowflake) |
| `eva_strip_think` | `EVA_STRIP_THINK` | `bool` | `False` | 💭 If set (present in env), strips 'think' steps from EVA output |
| `embedding_api` | `EMBEDDING_API` | `ModelAPI` | `HUGGINGFACE` | 🔌 API provider for embeddings (`OPENAI`, `HUGGINGFACE`, `OLLAMA`) |
| `llm_api` | `LLM_API` | `ModelAPI` | `OLLAMA` | 🤖 API provider for LLM (`OPENAI`, `HUGGINGFACE`, `OLLAMA`) |
| `max_research_loops` | `MAX_RESEARCH_LOOPS` | `int` | `3` | 🔁 Maximum number of research loops to perform |
| `llm_tool_model` | `LLM_TOOL_MODEL` | `str` | `smollm2:1.7b-instruct-q2_K` | 🛠️ LLM model for tool calling |
| `n_context_docs` | `N_CONTEXT_DOCS` | `int` | `2` | 📚 Number of context documents to retrieve for RAG |
| `search_permission` | `EVA_SEARCH_PERMISSION` | `str` | `no` | 🌐 Permission for search (`yes`, `no`, `ask`) |
| `db_persist` | `EVA_DB_PERSIST` | `str or None` | `None` | 💾 Path or flag for DB persistence |
| `eva_reinitialize` | `EVA_REINITIALIZE` | `bool` | `False` | 🔄 If true, reinitializes EVA DB |
| `thread_id` | `THREAD_ID` | `str` | `""` | 🧵 Thread ID for the current session |

- All fields can be set via environment variables (see [Pydantic BaseSettings docs](https://docs.pydantic.dev/latest/usage/settings/)).
- Types are enforced at runtime. Defaults are shown above.
- For advanced usage, see the `Configuration` class in `pstuts_rag/configuration.py`.

## 🎨 UI Customization & Theming

### Sepia Theme Implementation 🖼️

The application features a custom **sepia-toned color scheme** implemented via `public/theme.json` and Chainlit's theme configuration:

#### 📁 Theme Files
- **`public/theme.json`**: Defines the sepia color palette and theme variables
- **`.chainlit/config.toml`**: Configuration enabling the sepia theme as default

#### 🎨 Color Palette Design
Theme colors are defined in `theme.json` and applied through Chainlit's theming system. There is no custom CSS file; all theming is handled via JSON and Chainlit configuration.

#### ⚙️ Configuration Setup
```toml
# .chainlit/config.toml
[UI]
default_theme = "light"           # Set light theme as default
custom_theme = "/public/theme.json"  # Enable custom sepia theme
```

#### 🎯 Features
- **Responsive Design**: Adapts to both light and dark preferences
- **Accessibility**: Maintains sufficient contrast ratios in both themes
- **Visual Cohesion**: Unified sepia treatment across all UI elements
- **Performance**: JSON-based theme for minimal runtime overhead
- **User Control**: Native Chainlit theme switcher toggles between variants

The sepia theme creates a warm, nostalgic atmosphere perfect for Adobe Photoshop tutorials, giving the application a distinctive visual identity that stands out from standard blue/gray interfaces. 📸✨

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

## 🌊 Lazy Graph Initialization

The project uses a **lazy initialization pattern** for the LangGraph to avoid expensive compilation during module imports while maintaining compatibility with LangGraph Studio.

### 🔧 Implementation Pattern

```python
# In pstuts_rag/nodes.py
_compiled_graph = None

def graph(config: RunnableConfig = None):
    """Graph factory function for LangGraph Studio compatibility.
    
    This function provides lazy initialization of the graph and datastore,
    allowing the module to be imported without triggering compilation.
    LangGraph Studio requires this function to take exactly one RunnableConfig argument.
    
    Args:
        config: RunnableConfig (required by LangGraph Studio, but can be None)
    
    Returns:
        Compiled LangGraph instance
    """
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = graph_builder.compile()
        # Initialize datastore when graph is first accessed
        asyncio.run(datastore.from_json_globs(Configuration().transcript_glob))
    return _compiled_graph

def get_graph():
    """Convenience function to get the compiled graph without config argument."""
    return graph()
```

### 🎯 Benefits

- **Fast imports**: Module loading doesn't trigger graph compilation 🚀
- **LangGraph Studio compatibility**: Maintains expected `graph` variable for discovery 🛠️
- **On-demand initialization**: Graph and datastore only initialize when actually used ⚡
- **Memory efficiency**: Resources allocated only when needed 💾

### 📄 Studio Configuration

The `langgraph.json` file correctly references the factory function:
```json
{
    "graphs": {
        "enhanced_video_archive": "./pstuts_rag/pstuts_rag/nodes.py:graph"
    }
}
```

When LangGraph Studio accesses the `graph` function, it automatically triggers lazy initialization and provides the compiled graph instance. The factory function pattern ensures compatibility while maintaining performance benefits.

## 🏗️ Architecture Notes

- **Embedding models**: Uses custom fine-tuned `snowflake-arctic-embed-s-ft-pstuts` by default
- **Vector store**: Qdrant with semantic chunking for optimal retrieval
- **LLM**: GPT-4.1-mini for generation and routing
- **Web search**: Tavily integration targeting `helpx.adobe.com`
- **State management**: LangGraph for multi-agent coordination
- **Evaluation**: RAGAS framework for retrieval and generation metrics

## 🆕 Recent Refactors & Enhancements (Spring 2024)

### 🏗️ Modular App Structure & Async Initialization
- The main application (`app.py`) is now more modular and async-friendly! Initialization of the datastore, agent graph, and session state is handled with care for concurrency and user experience.
- The agent graph is now referenced as `ai_graph` (formerly `compiled_graph`) for clarity and onboarding ease.
- Chainlit session and callback management is improved, making it easier to hook into events and extend the app. 🚦

### 🤖 Robust API/Model Selection Logic
- All API/model selection (for LLMs and embeddings) is now centralized in `utils.py` via `get_chat_api` and `get_embeddings_api`.
- These functions robustly parse string input to the `ModelAPI` enum, so you can use any case or format (e.g., "openai", "OPENAI", "Ollama") and it will Just Work™.
- This eliminates a whole class of bugs from mismatched config strings! 🎉

### 🔍 Smarter Search Phrase Generation
- The search phrase generation logic (in `prompts.py` and node code) now uses previous queries and conversation history to generate unique, context-aware search phrases.
- This means less repetition, more relevance, and a more natural research workflow for the agents. 🧠✨

### ⚙️ Enhanced LLM API & Configuration
- The `Configuration` class (`configuration.py`) now supports robust environment variable overrides and easy conversion to/from `RunnableConfig`.
- All config parameters are logged and managed with dataclass fields, making debugging and onboarding a breeze.

### 🎨 Sepia Theme Update
- The UI now features a beautiful sepia color palette for a warm, inviting look (see above for details!).
- Theme files and configuration have been updated for seamless switching between light and dark sepia modes.
- Perfect for those late-night Photoshop tutorial sessions! ☕🖼️

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

#### 🛠️ Interactive Interrupt System Usage

**Environment Configuration:**
```bash
# Enable interactive prompts (default)
export EVA_SEARCH_PERMISSION="ask"

# Pre-approve all searches (autonomous mode)
export EVA_SEARCH_PERMISSION="yes" 

# Block all searches (local-only mode)  
export EVA_SEARCH_PERMISSION="no"
```

**Node Implementation Example:**
```python
# In search_help node (nodes.py)
decision = state["search_permission"]
if decision == YesNoAsk.ASK:
    # Pause execution and request user input
    response = interrupt(
        f"Do you allow Internet search for query '{query}'?"
        "Answer 'yes' will perform the search, any other answer will skip it."
    )
    
    # Parse user response  
    decision = YesNoAsk.YES if "yes" in response.strip() else YesNoAsk.NO
    
    # Update state and continue
    return Command(
        update={"search_permission": decision}, 
        goto=search_help.__name__
    )
```

**Runtime Behavior:**
```
User Query: "How do I use layer masks in Photoshop?"
System: "Do you allow Internet search for query 'How do I use layer masks in Photoshop?'? Answer 'yes' will perform the search, any other answer will skip it."
User: "yes"  
System: [Continues with web search + local RAG search]

User Query: "What are blend modes?"
System: "Do you allow Internet search for query 'What are blend modes?'? Answer 'yes' will perform the search, any other answer will skip it."  
User: "no"
System: [Skips web search, continues with local RAG only]
``` 

## 🛠️ Robust HTML Title Extraction

### `get_title_streaming(url)`

This function fetches the HTML from a URL and extracts the page title using all the most common conventions, in this order:

1. `<meta property="og:title" content="...">` (Open Graph, for social sharing)
2. `<meta name="twitter:title" content="...">` (Twitter Cards)
3. `<meta name="title" content="...">` (sometimes used for SEO)
4. `<title>...</title>` (the classic HTML title tag)

It returns the **first** found value as a string, or `None` if no title is found. All extraction is done with BeautifulSoup for maximum reliability and standards compliance.

#### Example usage:
```python
from pstuts_rag.utils import get_title_streaming
url = "https://example.com"
title = get_title_streaming(url)
print(title)  # Prints the best available title, or None
```

---

### 🥣 Requirements
- This function requires `beautifulsoup4` to be installed:
  ```bash
  pip install beautifulsoup4
  ```

---

> "A page by any other name would still be as sweet... but it's nice to get the right one!" 😄 

## 📝 Automatic TODO Extraction

This repo uses [`flake8-todos`](https://github.com/awestlake87/flake8-todos) to collect all TODO-style comments from Python files and writes them to a `TODO.md` file at the project root.

### How it works
- Run `uv run python scripts/generate_todo_md.py` to (re)generate `TODO.md`.
- A **manual pre-commit hook** is provided to automate this:
  1. Copy it into your git hooks:  
     `cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`
  2. On every commit, it will update `TODO.md` and stage it automatically.

### Why manual?
- This hook is not installed by default. You must opt-in by copying it yourself (see above).
- This keeps your workflow flexible and avoids surprises for new contributors.

### Example output
```
# 📝 TODOs in Codebase

- `pstuts_rag/agent.py:42`: TD003 TODO: Refactor this function
- `scripts/generate_todo_md.py:10`: TD002 FIXME: Handle edge case
```

Happy hacking! 🚀 
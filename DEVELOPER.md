# ✨ DEVELOPER.md ✨

This document is your trusty co-pilot for navigating the codebase of our amazing RAG application! 🚀

## ⚙️ Core Components

### `pstuts_rag/pstuts_rag/utils.py`

This module is a collection of utility functions that make our lives easier.

#### `get_chat_api(input: str) -> Type[ChatHuggingFace | ChatOpenAI | ChatOllama]`

Dynamically selects the appropriate chat model class (e.g., `ChatOpenAI`, `ChatHuggingFace`, `ChatOllama`) based on the `input` string.

*   **Previous Behavior**: Directly used the input string as a key for `ChatAPISelector`. This was prone to errors if the input string didn't exactly match a `ModelAPI` enum member's name.
*   **Current Behavior (Improved! 🥳)**: The `input` string is now robustly parsed into a `ModelAPI` enum member before being used as a key. This ensures that we correctly identify the desired API provider even if the input string's case or format varies slightly, as long as it's a valid `ModelAPI` value. For example, `"openai"` or `"OLLAMA"` will now correctly map to `ModelAPI.OPENAI` and `ModelAPI.OLLAMA` respectively.

This change makes the chat API selection more resilient and less error-prone! 🤓 
import os
import logging
from dataclasses import dataclass, fields
from typing import Any, Optional
from enum import Enum

from langchain_core.runnables import RunnableConfig


class ModelAPI(Enum):
    """Enum for supported embedding API providers."""

    OPENAI = "OPENAI"
    HUGGINGFACE = "HUGGINGFACE"
    OLLAMA = "OLLAMA"


@dataclass(kw_only=True)
class Configuration:
    """
    Configuration parameters for the application.

    Attributes:
        transcript_glob: Glob pattern for transcript JSON files (supports multiple files separated by ':')
        embedding_model: Name of the embedding model to use (default: custom fine-tuned snowflake model)
        embedding_api: API provider for embeddings (OPENAI or HUGGINGFACE)
        max_research_loops: Maximum number of research loops to perform
        llm_tool_model: Name of the LLM model to use for tool calling
        n_context_docs: Number of context documents to retrieve for RAG
    """

    eva_workflow_name: str = str(
        os.environ.get("EVA_WORKFLOW_NAME", "EVA_workflow")
    )

    eva_log_level: str = str(os.environ.get("EVA_LOG_LEVEL", "INFO")).upper()

    transcript_glob: str = str(
        os.environ.get("TRANSCRIPT_GLOB", "data/test.json")
    )

    embedding_model: str = str(
        os.environ.get(
            "EMBEDDING_MODEL", "mbudisic/snowflake-arctic-embed-s-ft-pstuts"
        )
    )

    embedding_api: ModelAPI = ModelAPI(
        os.environ.get("EMBEDDING_API", ModelAPI.HUGGINGFACE.value)
    )

    llm_api: ModelAPI = ModelAPI(
        os.environ.get("LLM_API", ModelAPI.OPENAI.value)
    )

    max_research_loops: int = int(os.environ.get("MAX_RESEARCH_LOOPS", "3"))

    llm_tool_model: str = str(
        os.environ.get("LLM_TOOL_MODEL", "smollm2:1.7b-instruct-q2_K")
    )
    n_context_docs: int = int(os.environ.get("N_CONTEXT_DOCS", "2"))

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"]
            if config and "configurable" in config
            else {}
        )
        # Map each dataclass field to environment variables or configurable values
        # Priority: environment variables > configurable dict values > field defaults
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

    def print(self, print_like_function=logging.info) -> None:
        """Log all configuration parameters using logging.debug."""
        print_like_function("Configuration parameters:")
        for field in fields(self):
            if field.init:
                value = getattr(self, field.name)
                print_like_function("  %s: %s", field.name, value)

import os
import logging
from typing import Any, Optional
from enum import Enum
from pydantic_settings import BaseSettings
from pydantic import Field

from langchain_core.runnables import RunnableConfig


class ModelAPI(Enum):
    """Enum for supported embedding API providers.

    Attributes:
        OPENAI: OpenAI API provider
        HUGGINGFACE: Hugging Face API provider
        OLLAMA: Ollama API provider
    """

    OPENAI = "OPENAI"
    HUGGINGFACE = "HUGGINGFACE"
    OLLAMA = "OLLAMA"


class Configuration(BaseSettings):
    """
    Configuration parameters for the application. All fields can be set via environment variables.
    """

    eva_workflow_name: str = Field(
        default_factory=lambda: os.environ.get(
            "EVA_WORKFLOW_NAME", "EVA_workflow"
        ),
        description="Name of the EVA workflow. Set via EVA_WORKFLOW_NAME.",
    )
    eva_log_level: str = Field(
        default_factory=lambda: os.environ.get(
            "EVA_LOG_LEVEL", "INFO"
        ).upper(),
        description="Logging level for EVA. Set via EVA_LOG_LEVEL.",
    )
    transcript_glob: str = Field(
        default_factory=lambda: os.environ.get(
            "TRANSCRIPT_GLOB", "data/test.json"
        ),
        description="Glob pattern for transcript JSON files (supports multiple files separated by ':'). Set via TRANSCRIPT_GLOB.",
    )
    embedding_model: str = Field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "mbudisic/snowflake-arctic-embed-s-ft-pstuts"
        ),
        description="Name of the embedding model to use (default: custom fine-tuned snowflake model). Set via EMBEDDING_MODEL.",
    )

    embedding_api: ModelAPI = Field(
        default_factory=lambda: ModelAPI(
            os.environ.get("EMBEDDING_API", ModelAPI.HUGGINGFACE.value)
        ),
        description="API provider for embeddings (OPENAI, HUGGINGFACE, or OLLAMA). Set via EMBEDDING_API.",
    )
    llm_api: ModelAPI = Field(
        default_factory=lambda: ModelAPI(
            os.environ.get("LLM_API", ModelAPI.OLLAMA.value)
        ),
        description="API provider for LLM (OPENAI, HUGGINGFACE, or OLLAMA). Set via LLM_API.",
    )
    max_research_loops: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_RESEARCH_LOOPS", "3")),
        description="Maximum number of research loops to perform. Set via MAX_RESEARCH_LOOPS.",
    )
    llm_tool_model: str = Field(
        default_factory=lambda: os.environ.get(
            "LLM_TOOL_MODEL", "smollm2:1.7b-instruct-q2_K"
        ),
        description="Name of the LLM model to use for tool calling. Set via LLM_TOOL_MODEL.",
    )
    n_context_docs: int = Field(
        default_factory=lambda: int(os.environ.get("N_CONTEXT_DOCS", "2")),
        description="Number of context documents to retrieve for RAG. Set via N_CONTEXT_DOCS.",
    )
    search_permission: str = Field(
        default_factory=lambda: os.environ.get("EVA_SEARCH_PERMISSION", "no"),
        description="Permission for search (yes/no). Set via EVA_SEARCH_PERMISSION.",
    )
    db_persist: Optional[str] = Field(
        default_factory=lambda: os.environ.get("EVA_DB_PERSIST", None),
        description="Path or flag for DB persistence. Set via EVA_DB_PERSIST.",
    )
    eva_reinitialize: bool = Field(
        default_factory=lambda: os.environ.get(
            "EVA_REINITIALIZE", "False"
        ).lower()
        in ("true", "1", "yes"),
        description="If true, reinitializes EVA DB. Set via EVA_REINITIALIZE.",
    )
    eva_strip_think: bool = Field(
        default_factory=lambda: os.environ.get(
            "EVA_STRIP_THINK", "True"
        ).lower()
        in ("true", "1", "yes"),
        description="If true (default) strips thinking tags from LLM responses. Set via EVA_STRIP_THINK.",
    )

    thread_id: str = Field(
        default="",
        description="Thread ID for the current session. Set via THREAD_ID.",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra env vars in .env/environment

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.

        Args:
            config: Optional RunnableConfig containing configurable parameters

        Returns:
            Configuration: New Configuration instance with values from config

        Note:
            Priority order: environment variables > configurable dict values > field defaults
        """
        configurable = (
            config["configurable"]
            if config and "configurable" in config
            else {}
        )
        # Map each field to environment variables or configurable values
        # Priority: environment variables > configurable dict values > field defaults
        values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.__fields__
        }
        logging.info("Configuration:\n%s", values)
        return cls(**{k: v for k, v in values.items() if v is not None})

    def print(self, print_like_function=logging.info) -> None:
        """Print all configuration parameters using the provided logging function.

        Args:
            print_like_function: Function to use for printing (defaults to logging.info)

        Returns:
            None
        """
        print_like_function("Configuration parameters:")
        for name, field in self.__fields__.items():
            value = getattr(self, name)
            print_like_function("  %s: %s", name, value)

    def to_runnable_config(self) -> RunnableConfig:
        """Convert Configuration instance to RunnableConfig format.

        Returns:
            RunnableConfig: Properly formatted configuration for LangGraph
        """
        configurable_dict = {}
        for name in self.__fields__:
            value = getattr(self, name)
            if value:
                configurable_dict[name] = value
        if self.thread_id:
            configurable_dict["thread_id"] = self.thread_id
        return RunnableConfig(configurable=configurable_dict)

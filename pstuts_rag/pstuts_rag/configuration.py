import os
from dataclasses import dataclass, fields
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """
    Configuration parameters for the application.

    Attributes:
        transcript_glob: Glob pattern for transcript JSON files (supports multiple files separated by ':')
        embedding_model: Name of the embedding model to use (default: custom fine-tuned snowflake model)
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

    max_research_loops: int = int(os.environ.get("MAX_RESEARCH_LOOPS", "3"))

    llm_tool_model: str = str(os.environ.get("LLM_TOOL_MODEL", "gpt-4.1-mini"))
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
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})

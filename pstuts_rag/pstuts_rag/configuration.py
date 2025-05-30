from dataclasses import dataclass


@dataclass
class Configuration:
    """
    Configuration parameters for the application.

    Attributes:
        filename: List of JSON file paths to load data from
        embedding_model: Name of the OpenAI embedding model to use
        n_context_docs: Number of context documents to retrieve
        tool_calling_model: Name of the OpenAI model to use for tool calling
    """

    filename = [f"data/{f}.json" for f in ["dev", "test", "train"]]
    embedding_model = "mbudisic/snowflake-arctic-embed-s-ft-pstuts"
    n_context_docs = 2
    tool_calling_model = "gpt-4.1-mini"

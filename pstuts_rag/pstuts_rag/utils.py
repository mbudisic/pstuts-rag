from typing import Dict, Type

from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings

from pstuts_rag.configuration import ModelAPI

# Chat model selector dictionary
"""
ChatAPISelector: Dictionary mapping ModelAPI enum values to their corresponding chat model classes.

This selector enables dynamic instantiation of chat models based on the configured
API provider. Supports OpenAI, HuggingFace, and Ollama chat model implementations.

Type:
    Dict[ModelAPI, Type[ChatHuggingFace | ChatOpenAI | ChatOllama]]

Keys:
    ModelAPI.HUGGINGFACE: Maps to ChatHuggingFace class
    ModelAPI.OPENAI: Maps to ChatOpenAI class  
    ModelAPI.OLLAMA: Maps to ChatOllama class

Example:
    >>> from pstuts_rag.configuration import ModelAPI
    >>> from pstuts_rag.utils import ChatAPISelector
    >>> 
    >>> # Get the appropriate chat model class
    >>> api_provider = ModelAPI.OPENAI
    >>> chat_class = ChatAPISelector[api_provider]
    >>> 
    >>> # Instantiate the chat model
    >>> chat_model = chat_class(model="gpt-3.5-turbo", temperature=0.7)
    >>> 
    >>> # Alternative usage with configuration
    >>> config = Configuration(llm_api=ModelAPI.OLLAMA)
    >>> chat_class = ChatAPISelector[config.llm_api]
    >>> chat_model = chat_class(model="llama2:7b")
"""
ChatAPISelector: Dict[
    ModelAPI, Type[ChatHuggingFace | ChatOpenAI | ChatOllama]
] = {
    ModelAPI.HUGGINGFACE: ChatHuggingFace,
    ModelAPI.OPENAI: ChatOpenAI,
    ModelAPI.OLLAMA: ChatOllama,
}

# Embeddings model selector dictionary
"""
EmbeddingsAPISelector: Dictionary mapping ModelAPI enum values to their corresponding embedding model classes.

This selector enables dynamic instantiation of embedding models based on the configured
API provider. Supports OpenAI, HuggingFace, and Ollama embedding implementations.

Type:
    Dict[ModelAPI, Type[HuggingFaceEmbeddings | OpenAIEmbeddings | OllamaEmbeddings]]

Keys:
    ModelAPI.HUGGINGFACE: Maps to HuggingFaceEmbeddings class
    ModelAPI.OPENAI: Maps to OpenAIEmbeddings class
    ModelAPI.OLLAMA: Maps to OllamaEmbeddings class

Example:
    >>> from pstuts_rag.configuration import ModelAPI
    >>> from pstuts_rag.utils import EmbeddingsAPISelector
    >>> 
    >>> # Get the appropriate embeddings model class
    >>> api_provider = ModelAPI.HUGGINGFACE
    >>> embeddings_class = EmbeddingsAPISelector[api_provider]
    >>> 
    >>> # Instantiate the embeddings model
    >>> embeddings = embeddings_class(
    ...     model_name="sentence-transformers/all-MiniLM-L6-v2"
    ... )
    >>> 
    >>> # Alternative usage with configuration
    >>> config = Configuration(embedding_api=ModelAPI.OPENAI)
    >>> embeddings_class = EmbeddingsAPISelector[config.embedding_api]
    >>> embeddings = embeddings_class(model="text-embedding-3-small")
"""
EmbeddingsAPISelector: Dict[
    ModelAPI, Type[HuggingFaceEmbeddings | OpenAIEmbeddings | OllamaEmbeddings]
] = {
    ModelAPI.HUGGINGFACE: HuggingFaceEmbeddings,
    ModelAPI.OPENAI: OpenAIEmbeddings,
    ModelAPI.OLLAMA: OllamaEmbeddings,
}

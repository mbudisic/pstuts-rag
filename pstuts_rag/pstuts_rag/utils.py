import logging
import re
from typing import Any, Dict, Iterator, List, Type

import requests
from bs4 import BeautifulSoup
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from pstuts_rag.configuration import ModelAPI

# Chat model selector dictionary
ChatAPISelector: Dict[
    ModelAPI, Type[ChatHuggingFace | ChatOpenAI | ChatOllama]
] = {
    ModelAPI.HUGGINGFACE: ChatHuggingFace,
    ModelAPI.OPENAI: ChatOpenAI,
    ModelAPI.OLLAMA: ChatOllama,
}


def get_chat_api(input: str):

    logging.info("LLM_API: %s", input)
    cls = ChatAPISelector[ModelAPI(input)]
    logging.info("LLM SELECTED: %s", cls)

    return cls


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


# Embeddings model selector dictionary
EmbeddingsAPISelector: Dict[
    ModelAPI, Type[HuggingFaceEmbeddings | OpenAIEmbeddings | OllamaEmbeddings]
] = {
    ModelAPI.HUGGINGFACE: HuggingFaceEmbeddings,
    ModelAPI.OPENAI: OpenAIEmbeddings,
    ModelAPI.OLLAMA: OllamaEmbeddings,
}


def get_embeddings_api(input: str):

    logging.info("EMBEDDINGS_API: %s", input)
    cls = EmbeddingsAPISelector[ModelAPI(input)]
    logging.info("EMBEDDINGS SELECTED: %s", cls)

    return cls


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


def flatten(lst: List[Any]):
    """
    Recursively flatten a nested list structure into a single-level generator.

    Takes a list that may contain nested lists and yields all elements
    in a flat sequence. Uses recursive generators to handle arbitrary
    nesting depth efficiently.

    Args:
        lst (List[Any]): The input list which may contain nested lists

    Yields:
        Any: Individual elements from the flattened list structure

    Example:
        >>> list(flatten([1, [2, 3], [4, [5, 6]]]))
        [1, 2, 3, 4, 5, 6]

        >>> list(flatten(['a', ['b', 'c'], 'd']))
        ['a', 'b', 'c', 'd']
    """
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def batch(iterable: List[Any], size: int = 16) -> Iterator[List[Any]]:
    """
    Batch an iterable into chunks of specified size.

    Yields successive chunks from the input iterable, each containing
    at most 'size' elements. Useful for processing large collections
    in manageable batches to avoid memory issues or API rate limits.

    Args:
        iterable (List[Any]): The input list to be batched
        size (int, optional): Maximum size of each batch. Defaults to 16.

    Yields:
        List[Any]: Successive batches of the input iterable

    Example:
        >>> list(batch([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def get_title_streaming(url):
    """
    Fetches the entire HTML content from the given URL and extracts the page title using common conventions:
    1. <meta property="og:title" content="...">
    2. <meta name="twitter:title" content="...">
    3. <meta name="title" content="...">
    4. <title>...</title>
    Returns the first found title as a string, or None if not found.
    """
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        # 1. Open Graph
        meta_og = soup.find("meta", attrs={"property": "og:title"})
        if meta_og and meta_og.has_attr("content"):
            title = meta_og["content"].strip()
            title = re.sub(r"\s+", " ", title)
            return title

        # 2. Twitter Card
        meta_twitter = soup.find("meta", attrs={"name": "twitter:title"})
        if meta_twitter and meta_twitter.has_attr("content"):
            title = meta_twitter["content"].strip()
            title = re.sub(r"\s+", " ", title)
            return title

        # 3. Meta name="title"
        meta_name = soup.find("meta", attrs={"name": "title"})
        if meta_name and meta_name.has_attr("content"):
            title = meta_name["content"].strip()
            title = re.sub(r"\s+", " ", title)
            return title

        # 4. <title>
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            title = re.sub(r"\s+", " ", title)
            return title

    except Exception as e:
        print(f"Error: {e}")
    return None


def get_unique(plain_list: list) -> list:
    unique_list = []
    for item in plain_list:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list

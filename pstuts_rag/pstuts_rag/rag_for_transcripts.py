import functools
import json
import asyncio
from operator import itemgetter
from typing import Any, Dict, Union, Optional, Callable
import logging
import re

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
    RunnableConfig,
    RunnableLambda,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama

from .datastore import DatastoreManager
from .prompts import RAG_PROMPT_TEMPLATES
from pstuts_rag.utils import ChatAPISelector
from pstuts_rag.configuration import Configuration, ModelAPI


def post_process_response(
    msg_dict: Dict[str, Any], config: RunnableConfig
) -> AIMessage:
    """Pack reference information into the AI message.

    Takes the generated answer and input context, formats references,
    and appends them to the message content.

    Args:
        msg_dict: Dictionary containing the answer and input

    Returns:
        AIMessage: Message with references appended
    """

    configurable = Configuration.from_runnable_config(config)

    answer: AIMessage = msg_dict["answer"]
    input = msg_dict["input"]

    # Extract relevant metadata from each document in the context
    reference_dicts = [
        {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}
        for doc in input["context"]
    ]
    references = str(json.dumps(reference_dicts, indent=2))

    text_w_references = (
        strip_think_tags(str(answer.content))
        if configurable.eva_strip_think
        else answer.content
    )
    # Only append references if the model provided a substantive answer
    if "I don't know" not in answer.content:
        text_w_references = "\n".join(
            [str(text_w_references), "**REFERENCES**", references]
        )

    # Create new message with references and preserve original context metadata
    output: AIMessage = answer.model_copy(
        update={
            "content": text_w_references,
            "additional_kwargs": {
                **answer.additional_kwargs,
                "context": input["context"],
                "question": input["question"],
            },
        }
    )

    return output


def strip_think_tags(input: str) -> str:
    """Removes everything between <think> and </think> tags (including tags)
    from the input string. If only <think> is present, it removes everything
    after it (including it.)

    Args:
        input (str): The input string potentially containing think tags

    Returns:
        str: The string with think tags and their content removed
    """
    # First, try to remove complete <think>...</think> blocks
    # Use non-greedy matching to handle multiple blocks correctly
    result = re.sub(r"<think>.*?</think>", "", input, flags=re.DOTALL)

    # Then, handle case where only opening <think> tag exists
    # Remove everything from <think> to the end of the string
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)

    return result


def create_transcript_rag_chain(
    datastore: DatastoreManager,
    config: Union[RunnableConfig, Configuration] = Configuration(),
) -> Runnable:
    """Create a Retrieval-Augmented Generation (RAG) chain for video transcript search.

    This function constructs a complete RAG pipeline that:
    1. Takes a user question as input
    2. Retrieves relevant video transcript chunks from the datastore
    3. Generates an answer using an LLM with the retrieved context
    4. Packages the response with reference information

    Args:
        datastore: The DatastoreManager containing video transcript embeddings
        config: Configuration object or RunnableConfig with model and retrieval settings

    Returns:
        Runnable: A LangChain runnable that processes questions and returns
                 answers with embedded references to source video segments
    """

    # Handle both Configuration objects and RunnableConfig dictionaries
    configurable = (
        config
        if isinstance(config, Configuration)
        else Configuration.from_runnable_config(config)
    )

    # Select the appropriate chat model class based on configuration
    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)

    llm = cls(model=configurable.llm_tool_model)

    # Create the answer generation chain using prompt templates
    answer_chain = (
        ChatPromptTemplate.from_messages(list(RAG_PROMPT_TEMPLATES.items()))
        | llm
    )

    # Build the complete RAG chain with the following flow:
    # question -> parallel(context_retrieval, question_passthrough) -> llm_answer -> pack_references
    rag_chain = (
        itemgetter("question")  # Extract question from input dict
        | RunnableParallel(  # Run context retrieval and question passing in parallel
            context=datastore.get_retriever(
                n_context_docs=configurable.n_context_docs
            ),
            question=RunnablePassthrough(),  # Pass question unchanged
        )
        | {  # Prepare input dict for final processing
            "input": RunnablePassthrough(),  # Contains both context and question
            "answer": answer_chain,  # Generate answer using retrieved context
        }
        | RunnableLambda(
            post_process_response
        )  # wrapping in runnable Lambda to pass config
    )

    return rag_chain

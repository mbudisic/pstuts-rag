import json
import asyncio
from operator import itemgetter
from typing import Any, Dict, Union, Optional, Callable
import logging

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableParallel,
    RunnablePassthrough,
    RunnableConfig,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama

from .datastore import DatastoreManager
from .prompts import RAG_PROMPT_TEMPLATES

from pstuts_rag.configuration import Configuration, ModelAPI


def pack_references(msg_dict: Dict[str, Any]) -> AIMessage:
    """Pack reference information into the AI message.

    Takes the generated answer and input context, formats references,
    and appends them to the message content.

    Args:
        msg_dict: Dictionary containing the answer and input

    Returns:
        AIMessage: Message with references appended
    """
    answer: AIMessage = msg_dict["answer"]
    input = msg_dict["input"]

    reference_dicts = [
        {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}
        for doc in input["context"]
    ]
    references = str(json.dumps(reference_dicts, indent=2))

    text_w_references = answer.content
    if "I don't know" not in answer.content:
        text_w_references = "\n".join(
            [str(text_w_references), "**REFERENCES**", references]
        )

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


def retrieve_videos(
    datastore: DatastoreManager,
    config: Union[RunnableConfig, Configuration] = Configuration(),
) -> Runnable:

    configurable = (
        config
        if isinstance(config, Configuration)
        else Configuration.from_runnable_config(config)
    )

    cls = {
        ModelAPI.HUGGINGFACE: ChatHuggingFace,
        ModelAPI.OPENAI: ChatOpenAI,
        ModelAPI.OLLAMA: ChatOllama,
    }.get(configurable.llm_api, ChatOpenAI)

    llm = cls(model=configurable.llm_tool_model)

    answer_chain = (
        ChatPromptTemplate.from_messages(list(RAG_PROMPT_TEMPLATES.items()))
        | llm
    )

    rag_chain = (
        itemgetter("question")
        | RunnableParallel(
            context=datastore.get_retriever(
                n_context_docs=configurable.n_context_docs
            ),
            question=RunnablePassthrough(),
        )
        | {
            "input": RunnablePassthrough(),
            "answer": answer_chain,
        }
        | pack_references
    )

    return rag_chain


def startup(
    config=Configuration(),
    callback_on_loading_complete: Optional[Callable] = None,
):
    """
    Initialize the application with optional loading completion callback.

    Args:
        config: Configuration object with application settings
        on_loading_complete: Optional callback (sync or async) to call when
                           datastore loading completes

    Returns:
        DatastoreManager: The initialized datastore manager
    """

    ### PROCESS THE CONFIGURATION
    log_level = getattr(logging, config.eva_log_level, logging.INFO)
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    ### CREATE THE DATABASE

    datastore = DatastoreManager()
    if callback_on_loading_complete:
        datastore.add_completion_callback(callback_on_loading_complete)

    ### START DATABASE POPULATION

    globs = [str(g) for g in config.transcript_glob.split(":")]

    # # Add custom callback if provided, otherwise use default logging
    # if on_loading_complete:
    #     datastore.add_completion_callback(on_loading_complete)
    # else:
    #     # Default callback for logging
    #     def default_logging_callback():
    #         logging.info("🎉 Datastore loading completed!")

    #     datastore.add_completion_callback(default_logging_callback)

    asyncio.create_task(datastore.from_json_globs(globs))

    return datastore

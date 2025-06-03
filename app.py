from pstuts_rag.configuration import Configuration
import requests
import asyncio
import json
import os
import getpass
from typing import List, Tuple
import re

import chainlit as cl
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.embeddings import Embeddings
from langgraph.checkpoint.memory import MemorySaver


from langchain_core.messages import HumanMessage, BaseMessage


from pstuts_rag.configuration import Configuration
from pstuts_rag.datastore import DatastoreManager
from pstuts_rag.rag_for_transcripts import create_transcript_rag_chain
from pstuts_rag.nodes import initialize

import nest_asyncio
from uuid import uuid4

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Apply nested asyncio to enable nested event loops
nest_asyncio.apply()

# Generate a unique ID for this application instance
unique_id = uuid4().hex[0:8]

VIDEOARCHIVE = "VideoArchiveSearch"
ADOBEHELP = "AdobeHelp"


def set_api_key_if_not_present(key_name, prompt_message=""):
    """
    Sets an API key in the environment if it's not already present.

    Args:
        key_name: Name of the environment variable to set
        prompt_message: Custom prompt message for getpass (defaults to key_name)
    """
    if len(prompt_message) == 0:
        prompt_message = key_name
    if key_name not in os.environ or not os.environ[key_name]:
        os.environ[key_name] = getpass.getpass(prompt_message)


class ApplicationState:
    """
    Maintains the state of the application and its components.

    Attributes:
        embeddings: Embeddings model for vector operations
        docs: List of loaded documents
        qdrant_client: Client for Qdrant vector database
        vector_store: Vector store for document retrieval
        datastore_manager: Manager for data storage and retrieval
        rag_factory: Factory for creating RAG chains
        llm: Language model instance
        rag_chain: Retrieval-augmented generation chain
        ai_graph: Compiled AI agent graph
        ai_graph_sketch: State graph for AI agent orchestration
        tasks: List of asyncio tasks
        hasLoaded: Event to track when loading is complete
        pointsLoaded: Number of data points loaded into the database
    """

    config: Configuration = Configuration()
    compiled_graph = None
    datastore: DatastoreManager = None
    checkpointer = MemorySaver()

    def __init__(self) -> None:
        """
        Initialize the application state and set up environment variables.
        """
        load_dotenv()
        set_api_key_if_not_present("OPENAI_API_KEY")
        set_api_key_if_not_present("TAVILY_API_KEY")
        # os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = (
            f"AIE - MBUDISIC - HF - CERT - {unique_id}"
        )
        set_api_key_if_not_present("LANGCHAIN_API_KEY")


# Initialize global application state
_app_state = ApplicationState()


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the application when a new chat session starts.

    Sets up the language model, vector database components, and spawns tasks
    for database population and graph building.
    """
    global _app_state

    # Initialize datastore using asyncio.to_thread to avoid blocking
    initialize_datastore: bool = _app_state.datastore is None or (
        isinstance(_app_state.datastore, DatastoreManager)
        and _app_state.datastore.count_docs() == 0
    )
    if initialize_datastore:
        _app_state.datastore = await asyncio.to_thread(
            lambda: DatastoreManager(
                config=_app_state.config
            ).add_completion_callback(
                lambda: cl.run_sync(
                    cl.Message(content="Datastore loading completed.").send()
                )
            )
        )

    # Initialize and compile graph synchronously (blocking as intended)
    if _app_state.compiled_graph is None:
        _app_state.datastore, graph_builder = initialize(_app_state.datastore)
        _app_state.compiled_graph = graph_builder.compile(
            checkpointer=_app_state.checkpointer
        )

    # Start datastore population as background task (non-blocking)
    if initialize_datastore:
        asyncio.create_task(
            _app_state.datastore.from_json_globs(
                _app_state.config.transcript_glob
            )
        )


# def process_response(
#     response_message: BaseMessage,
# ) -> Tuple[str, List[cl.Message]]:
#     """
#     Processes a response from the AI agents.

#     Extracts the main text and video references from the response,
#     and creates message elements for displaying video content.

#     Args:
#         response: Response object from the AI agent

#     Returns:
#         Tuple containing the text response and a list of message elements with video references
#     """
#     streamed_text = f"[_from: {response_message.name}_]\n"
#     msg_references = []

#     if response_message.name == VIDEOARCHIVE:
#         text, references = pstuts_rag.rag.RAGChainFactory.unpack_references(
#             str(response_message.content)
#         )
#         streamed_text += text

#         if len(references) > 0:
#             references = json.loads(references)
#             print(references)

#             for ref in references:
#                 msg_references.append(
#                     cl.Message(
#                         content=(
#                             f"Watch {ref['title']} from timestamp "
#                             f"{round(ref['start'] // 60)}m:{round(ref['start'] % 60)}s"
#                         ),
#                         elements=[
#                             cl.Video(
#                                 name=ref["title"],
#                                 url=f"{ref['source']}#t={ref['start']}",
#                                 display="side",
#                             )
#                         ],
#                     )
#                 )
#     else:
#         streamed_text += str(response_message.content)

#         # Find all URLs in the content
#         urls = re.findall(
#             r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[/\w\.-=&%]*)?",
#             str(response_message.content),
#         )
#         print(urls)
#         links = []
#         # Create a list of unique URLs
#         for idx, u in enumerate(list(set(urls))):

#             url = "https://api.microlink.io"
#             params = {
#                 "url": u,
#                 "screenshot": True,
#             }

#             payload = requests.get(url, params)

#             if payload:
#                 print(f"Successful screenshot\n{payload.json()}")
#                 links.append(
#                     cl.Image(
#                         name=f"Website {idx} Preview: {u}",
#                         display="side",  # Show in the sidebar
#                         url=payload.json()["data"]["screenshot"]["url"],
#                     )
#                 )

#         print(links)
#         msg_references.append(
#             cl.Message(
#                 content="\n".join([l.url for l in links]), elements=links
#             )
#         )

#     return streamed_text, msg_references


@cl.on_message
async def main(user_cl_message: cl.Message):
    """
    Processes incoming user messages and sends responses.

    Streams the AI agent's response, processes it to extract text and video references,
    and sends the content back to the user's chat interface.

    Args:
        message: User's input message
    """
    # for s in app_state.ai_graph.stream(
    #     user_cl_message.content, {"recursion_limit": 20}
    # ):
    #     if "__end__" not in s and "supervisor" not in s.keys():
    #         for [node_type, node_response] in s.items():
    #             print(f"Processing {node_type} messages")
    #             for node_message in node_response["messages"]:
    #                 print(f"Message {node_message}")
    #                 msg = cl.Message(content="")
    #                 text, references = process_response(node_message)
    #                 for token in [char for char in text]:
    #                     await msg.stream_token(token)
    #                 await msg.send()
    #                 for m in references:
    #                     await m.send()


if __name__ == "__main__":
    main()

import requests
import asyncio
import json
import os
import getpass
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.messages import HumanMessage, BaseMessage
import langgraph.graph

from pstuts_rag.agents import PsTutsTeamState, create_team_supervisor
from pstuts_rag.agent_tavily import create_tavily_node

import pstuts_rag.datastore
import pstuts_rag.rag

from pstuts_rag.agent_rag import create_rag_node

from pstuts_rag.loader import load_json_files
from pstuts_rag.prompt_templates import SUPERVISOR_SYSTEM

import nest_asyncio
from uuid import uuid4

from sentence_transformers import SentenceTransformer
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

# Apply nested asyncio to enable nested event loops
nest_asyncio.apply()

# Generate a unique ID for this application instance
unique_id = uuid4().hex[0:8]

VIDEOARCHIVE = "VideoArchiveSearch"
ADOBEHELP = "AdobeHelp"


@dataclass
class ApplicationParameters:
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
        embeddings: OpenAI embeddings model for vector operations
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

    embeddings: Embeddings = None
    docs: List[Document] = []
    qdrant_client: QdrantClient = None
    vector_store: QdrantVectorStore = None
    datastore_manager: pstuts_rag.datastore.DatastoreManager
    rag: pstuts_rag.rag.RAGChainInstance
    llm: BaseChatModel
    rag_chain: Runnable

    ai_graph: Runnable
    ai_graph_sketch: langgraph.graph.StateGraph

    tasks: List[asyncio.Task] = []

    hasLoaded: asyncio.Event = asyncio.Event()
    pointsLoaded: int = 0

    def __init__(self) -> None:
        """
        Initialize the application state and set up environment variables.
        """
        load_dotenv()
        set_api_key_if_not_present("OPENAI_API_KEY")
        set_api_key_if_not_present("TAVILY_API_KEY")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = (
            f"AIE - MBUDISIC - HF - CERT - {unique_id}"
        )
        set_api_key_if_not_present("LANGCHAIN_API_KEY")


# Initialize global application state
app_state = ApplicationState()
params = ApplicationParameters()
ai_state = PsTutsTeamState(
    messages=[],
    team_members=[VIDEOARCHIVE, ADOBEHELP],
    next="START",
)


async def fill_the_db(
    state: ApplicationState,
):
    """
    Populates the vector database with document data if it's empty.

    Args:
        state: Application state containing the datastore manager

    Returns:
        0 if database already has documents, otherwise None
    """
    data: List[Dict[str, Any]] = await load_json_files(params.filename)

    _ = await state.rag.build_chain(data)
    await cl.Message(
        content=f"✅ The database has been loaded with {state.rag.pointsLoaded} elements!"
    ).send()


async def build_the_graph(current_state: ApplicationState):
    """
    Builds the agent graph for routing user queries.

    Creates the necessary nodes (Adobe help, RAG search, supervisor), defines their
    connections, and compiles the graph into a runnable chain.

    Args:
        current_state: Current application state with required components
    """
    adobe_help_node, _, _ = create_tavily_node(
        llm=app_state.llm, name=ADOBEHELP
    )

    rag_node, _ = create_rag_node(
        rag_chain=current_state.rag.rag_chain,
        name=VIDEOARCHIVE,
    )

    supervisor_agent = create_team_supervisor(
        current_state.llm,
        SUPERVISOR_SYSTEM,
        [VIDEOARCHIVE, ADOBEHELP],
    )

    ai_graph = langgraph.graph.StateGraph(PsTutsTeamState)

    ai_graph.add_node(VIDEOARCHIVE, rag_node)
    ai_graph.add_node(ADOBEHELP, adobe_help_node)
    ai_graph.add_node("supervisor", supervisor_agent)

    edges = [
        [VIDEOARCHIVE, "supervisor"],
        [ADOBEHELP, "supervisor"],
    ]

    [ai_graph.add_edge(*p) for p in edges]

    ai_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            VIDEOARCHIVE: VIDEOARCHIVE,
            ADOBEHELP: ADOBEHELP,
            "FINISH": langgraph.graph.END,
        },
    )

    ai_graph.set_entry_point("supervisor")
    app_state.ai_graph_sketch = ai_graph
    app_state.ai_graph = enter_chain | ai_graph.compile()


async def initialize():

    await fill_the_db(app_state)
    await build_the_graph(app_state)


def enter_chain(message: str):
    """
    Entry point for the agent graph chain.

    Transforms a user message into the state format expected by the agent graph.

    Args:
        message: User's input message

    Returns:
        Dictionary with the message and team members information
    """
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": [VIDEOARCHIVE, ADOBEHELP],
    }
    return results


@cl.on_chat_start
async def on_chat_start():
    """
    Initializes the application when a new chat session starts.

    Sets up the language model, vector database components, and spawns tasks
    for database population and graph building.
    """
    app_state.llm = ChatOpenAI(model=params.tool_calling_model, temperature=0)
    app_state.qdrant_client = QdrantClient(":memory:")
    app_state.embeddings = SentenceTransformer(params.embedding_model)

    app_state.rag = pstuts_rag.rag.RAGChainInstance(
        name="deployed",
        qdrant_client=app_state.qdrant_client,
        llm=app_state.llm,
        embeddings=app_state.embeddings,
    )

    app_state.tasks.append(asyncio.create_task(initialize()))


def process_response(
    response_message: BaseMessage,
) -> Tuple[str, List[cl.Message]]:
    """
    Processes a response from the AI agents.

    Extracts the main text and video references from the response,
    and creates message elements for displaying video content.

    Args:
        response: Response object from the AI agent

    Returns:
        Tuple containing the text response and a list of message elements with video references
    """
    streamed_text = f"[_from: {response_message.name}_]\n"
    msg_references = []

    if response_message.name == VIDEOARCHIVE:
        text, references = pstuts_rag.rag.RAGChainFactory.unpack_references(
            str(response_message.content)
        )
        streamed_text += text

        if len(references) > 0:
            references = json.loads(references)
            print(references)

            for ref in references:
                msg_references.append(
                    cl.Message(
                        content=(
                            f"Watch {ref['title']} from timestamp "
                            f"{round(ref['start'] // 60)}m:{round(ref['start'] % 60)}s"
                        ),
                        elements=[
                            cl.Video(
                                name=ref["title"],
                                url=f"{ref['source']}#t={ref['start']}",
                                display="side",
                            )
                        ],
                    )
                )
    else:
        streamed_text += str(response_message.content)

        # Find all URLs in the content
        urls = re.findall(
            r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[/\w\.-]*(?:\?[/\w\.-=&%]*)?",
            str(response_message.content),
        )
        print(urls)
        links = []
        # Create a list of unique URLs
        for idx, u in enumerate(list(set(urls))):

            url = "https://api.microlink.io"
            params = {
                "url": u,
                "screenshot": True,
            }

            payload = requests.get(url, params)

            if payload:
                print(f"Successful screenshot\n{payload.json()}")
                links.append(
                    cl.Image(
                        name=f"Website {idx} Preview: {u}",
                        display="side",  # Show in the sidebar
                        url=payload.json()["data"]["screenshot"]["url"],
                    )
                )

        print(links)
        msg_references.append(
            cl.Message(
                content="\n".join([l.url for l in links]), elements=links
            )
        )

    return streamed_text, msg_references


@cl.on_message
async def main(user_cl_message: cl.Message):
    """
    Processes incoming user messages and sends responses.

    Streams the AI agent's response, processes it to extract text and video references,
    and sends the content back to the user's chat interface.

    Args:
        message: User's input message
    """
    for s in app_state.ai_graph.stream(
        user_cl_message.content, {"recursion_limit": 20}
    ):
        if "__end__" not in s and "supervisor" not in s.keys():
            for [node_type, node_response] in s.items():
                print(f"Processing {node_type} messages")
                for node_message in node_response["messages"]:
                    print(f"Message {node_message}")
                    msg = cl.Message(content="")
                    text, references = process_response(node_message)
                    for token in [char for char in text]:
                        await msg.stream_token(token)
                    await msg.send()
                    for m in references:
                        await m.send()


if __name__ == "__main__":
    main()

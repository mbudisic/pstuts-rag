import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

import pstuts_rag.datastore
import pstuts_rag.rag
from pstuts_rag.datastore import load_json_files


@dataclass
class ApplicationParameters:
    filename = [f"data/{f}.json" for f in ["dev"]]
    embedding_model = "text-embedding-3-small"
    n_context_docs = 2
    llm_model = "gpt-4.1-mini"


def set_api_key_if_not_present(key_name, prompt_message=""):
    if len(prompt_message) == 0:
        prompt_message = key_name
    if key_name not in os.environ or not os.environ[key_name]:
        os.environ[key_name] = getpass.getpass(prompt_message)


class ApplicationState:
    embeddings: OpenAIEmbeddings = None
    docs: List[Document] = []
    qdrant_client: QdrantClient = None
    vector_store: QdrantVectorStore = None
    datastore_manager: pstuts_rag.datastore.DatastoreManager
    rag_factory: pstuts_rag.rag.RAGChainFactory
    llm: BaseChatModel
    rag_chain: Runnable

    hasLoaded: asyncio.Event = asyncio.Event()
    pointsLoaded: int = 0

    def __init__(self) -> None:
        load_dotenv()
        set_api_key_if_not_present("OPENAI_API_KEY")


state = ApplicationState()
params = ApplicationParameters()


async def fill_the_db():
    if state.datastore_manager.count_docs() == 0:
        data: List[Dict[str, Any]] = await load_json_files(params.filename)
        state.pointsLoaded = await state.datastore_manager.populate_database(
            raw_docs=data
        )
        await cl.Message(
            content=f"✅ The database has been loaded with {state.pointsLoaded} elements!"
        ).send()


async def build_the_chain():
    state.rag_factory = pstuts_rag.rag.RAGChainFactory(
        retriever=state.datastore_manager.get_retriever()
    )
    state.llm = ChatOpenAI(model=params.llm_model, temperature=0)
    state.rag_chain = state.rag_factory.get_rag_chain(state.llm)
    pass


@cl.on_chat_start
async def on_chat_start():
    state.qdrant_client = QdrantClient(":memory:")

    state.datastore_manager = pstuts_rag.datastore.DatastoreManager(
        qdrant_client=state.qdrant_client, name="local_test"
    )
    asyncio.run(main=fill_the_db())
    asyncio.run(main=build_the_chain())


@cl.on_message
async def main(message: cl.Message):
    # Send a response back to the user
    msg = cl.Message(content="")
    response = await state.rag_chain.ainvoke({"question": message.content})

    text, references = pstuts_rag.rag.RAGChainFactory.unpack_references(
        response.content
    )
    if isinstance(text, str):
        for token in [char for char in text]:
            await msg.stream_token(token)

    await msg.send()

    references = json.loads(references)
    print(references)

    msg_references = [
        (
            f"Watch {ref["title"]} from timestamp "
            f"{round(ref["start"] // 60)}m:{round(ref["start"] % 60)}s",
            cl.Video(
                name=ref["title"],
                url=f"{ref["source"]}#t={ref["start"]}",
                display="side",
            ),
        )
        for ref in references
    ]
    await cl.Message(content="Related videos").send()
    for e in msg_references:
        await cl.Message(content=e[0], elements=[e[1]]).send()


if __name__ == "__main__":
    main()

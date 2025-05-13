import asyncio
from typing import List
import chainlit as cl
import json
import os

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain_qdrant import QdrantVectorStore
from pstuts_rag.rag import RAGChainFactory, RetrieverFactory
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dataclasses import dataclass

import pstuts_rag.rag


@dataclass
class ApplicationParameters:
    filename = "data/test.json"
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
    qdrantclient: QdrantClient = None
    vectorstore: QdrantVectorStore = None
    retriever_factory: pstuts_rag.rag.RetrieverFactory
    rag_factory: pstuts_rag.rag.RAGChainFactory

    def __init__(self) -> None:
        load_dotenv()
        set_api_key_if_not_present("OPENAI_API_KEY")


state = ApplicationState()
params = ApplicationParameters()


@cl.on_chat_start
async def on_chat_start():
    state.client = QdrantClient(":memory:")

    state.retriever_factory = pstuts_rag.rag.RetrieverFactory(
        qdrant_client=state.client, name="local_test"
    )
    if state.retriever_factory.count_docs() == 0:
        data: List[Dict[str, Any]] = json.load(open(params.filename, "rb"))
        asyncio.run(main=state.retriever_factory.aadd_docs(raw_docs=data))
    state.rag_factory = pstuts_rag.rag.RAGChainFactory(
        retriever=state.retriever_factory.get_retriever()
    )
    state.llm = ChatOpenAI(model=params.llm_model, temperature=0)
    state.rag_chain = state.rag_factory.get_rag_chain(state.llm)


@cl.on_message
async def main(message: cl.Message):
    # Send a response back to the user

    v = await state.rag_chain.ainvoke(message.content)

    await cl.Message(content=v.content).send()


if __name__ == "__main__":
    main()

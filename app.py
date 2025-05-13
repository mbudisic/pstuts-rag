from typing import List
import chainlit as cl
import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from dataclasses import dataclass

import pstuts_rag.datastore


@dataclass
class ApplicationParameters:
    filename = "data/test.json"
    embedding_model = "text-embedding-3-small"


class ApplicationState:
    embeddings: OpenAIEmbeddings = None
    docs: List[Document] = []
    qdrantclient: QdrantClient = None
    vectorstore: QdrantVectorStore = None
    n_context_docs = 2
    retriever = None


state = ApplicationState()


@cl.on_chat_start
async def on_chat_start():
    params = ApplicationParameters()

    await cl.Message(content=f"Loading file {params.filename}").send()
    data = json.load(open(params.filename, "rb"))

    state.embeddings = OpenAIEmbeddings(model=params.embedding_model)
    state.docs = pstuts_rag.datastore.transcripts_load(data, state.embeddings)
    await cl.Message(
        content=f"Loaded {len(state.docs)} chunks from file {params.filename}."
    ).send()

    state.qdrantclient = QdrantClient(":memory:")

    state.vectorstore = pstuts_rag.datastore.initialize_vectorstore(
        client=state.qdrantclient,
        collection_name=f"{params.filename}_qdrant",
        embeddings=state.embeddings,
    )

    _ = state.vectorstore.add_documents(documents=state.docs)
    state.retriever = state.vectorstore.as_retriever(
        search_kwargs={"k": state.n_context_docs}
    )


@cl.on_message
async def main(message: cl.Message):
    # Send a response back to the user

    await cl.Message(content=f"Hello! You said: {message.content}").send()


if __name__ == "__main__":
    main()

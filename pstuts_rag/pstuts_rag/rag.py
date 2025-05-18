"""RAG (Retrieval Augmented Generation) module for PsTuts.

This module provides the core RAG functionality, including:
- RetrieverFactory: Creates and manages vector store retrievers
- RAGChainFactory: Builds RAG chains using LangChain components
"""

import json
import re

from operator import itemgetter
from typing import Any, Dict, List, Tuple
import logging


from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore

from .prompt_templates import RAG_PROMPT_TEMPLATES

from .datastore import DatastoreManager


class RAGChainFactory:
    """Factory class for creating RAG (Retrieval Augmented Generation) chains.

    This class builds LangChain RAG chains that:
    1. Take a user query
    2. Retrieve relevant documents
    3. Generate an answer based on those documents
    4. Append reference metadata to the answer

    Attributes:
        format_query: Runnable to extract the question from input
        retriever: Vector store retriever for document retrieval
        add_context_to_query: Runnable for adding context to queries
        prompt_template: Template for generating prompts to the LLM
        answer_chain: Chain for generating answers
        llm: Language model for answer generation
        rag_chain: The complete RAG chain
    """

    format_query = RunnableLambda(itemgetter("question"))
    retriever: VectorStoreRetriever
    add_context_to_query: Runnable
    prompt_template: Runnable
    answer_chain: Runnable
    llm: ChatOpenAI

    rag_chain: Runnable

    @staticmethod
    def compile_references(context: List[Document]) -> str:
        """Compile reference metadata from context documents.

        Args:
            context: List of Document objects with metadata

        Returns:
            str: JSON string of formatted reference metadata
        """
        references = [
            {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}
            for doc in context
        ]
        return str(json.dumps(references, indent=2))

    @staticmethod
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

        references: str = RAGChainFactory.compile_references(
            context=input["context"]
        )

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

    @staticmethod
    def unpack_references(content: str) -> Tuple[str, str]:
        parts = re.split(r"\*\*REFERENCES\*\*\s*", content, maxsplit=1)

        if len(parts) == 2:
            text = parts[0].rstrip()
            references = parts[1].lstrip()
            return text, references

        else:
            return content, ""

    def __init__(
        self,
        retriever: VectorStoreRetriever,
    ) -> None:
        """Initialize the RAGChainFactory.

        Args:
            retriever: Vector store retriever for document retrieval
        """
        self.retriever = retriever

        self.prepare_query = {
            "context": retriever,
            "question": RunnablePassthrough(),
        }

        self.prompt_template = ChatPromptTemplate.from_messages(
            list(RAG_PROMPT_TEMPLATES.items())
        )

        print(repr)

    def get_rag_chain(
        self,
        llm: BaseChatModel = ChatOpenAI(model="gpt-4.1-mini", temperature=0),
    ) -> Runnable:
        """Build and return the complete RAG chain.

        Constructs a chain that processes queries, retrieves relevant
        documents, generates an answer, and appends reference information.

        Args:
            llm: Language model to use for answer generation

        Returns:
            Runnable: The complete RAG chain
        """
        self.answer_chain = self.prompt_template | llm
        self.rag_chain = (
            self.format_query
            | self.prepare_query
            | {"input": RunnablePassthrough(), "answer": self.answer_chain}
            | self.pack_references
        )

        return self.rag_chain


class RAGChainInstance:
    """
    A class that encapsulates a Retrieval-Augmented Generation (RAG) chain.
    It has been abstracted from app_simple_rag.py but implemented in not-fully-
    async manner.

    This class manages the components needed for a RAG system, including
    embeddings, vector store, document storage, and the chain itself. It
    provides methods to initialize and build the RAG chain from JSON data.

    Attributes:
        embeddings (Embeddings): The embedding model used to convert text to vectors.
        docs (List[Document]): List of documents to be processed.
        qdrant_client (QdrantClient): Client for interacting with Qdrant vector database.
        vector_store (QdrantVectorStore): Vector store for document embeddings.
        datastore_manager (DatastoreManager): Manager for document storage and retrieval.
        rag_factory (RAGChainFactory): Factory for creating RAG chains.
        llm (BaseChatModel): Language model used for generating responses.
        rag_chain (Runnable): The assembled RAG chain.
        name (str): Identifier for this RAG chain instance.
        pointsLoaded (int): Number of data points loaded into the vector store.
    """

    embeddings: Embeddings = None
    docs: List[Document] = []
    qdrant_client: QdrantClient = None
    vector_store: QdrantVectorStore = None
    datastore_manager: DatastoreManager
    rag_factory: RAGChainFactory
    llm: BaseChatModel
    rag_chain: Runnable | None = None
    name: str

    pointsLoaded: int = 0

    def __init__(self, name, qdrant_client, llm, embeddings) -> None:
        """
        Initialize a new RAG chain instance.

        Args:
            name (str): Identifier for this RAG chain instance.
            qdrant_client (QdrantClient): Client for Qdrant vector database.
            llm (BaseChatModel): Language model for response generation.
            embeddings (Embeddings): Embedding model for text vectorization.
        """
        self.name = name
        self.qdrant_client = qdrant_client
        self.llm = llm
        self.embeddings = embeddings

    async def build_chain(
        self, json_payload: List[Dict[str, Any]]
    ) -> Runnable:
        """
        Build the RAG chain using the provided JSON data.

        This method initializes the datastore manager, populates the database if empty,
        creates the RAG factory, and assembles the final RAG chain.

        Args:
            json_payload (List[Dict[str,Any]]): List of JSON documents to be processed.

        Returns:
            Runnable: The assembled RAG chain ready for invocation.
        """

        self.datastore_manager = DatastoreManager(
            qdrant_client=self.qdrant_client, name=self.name
        )
        if self.datastore_manager.count_docs() == 0:
            self.pointsLoaded = await self.datastore_manager.populate_database(
                raw_docs=json_payload
            )
            logging.info(
                "JSON payload resulted in %d vectordb points.",
                self.pointsLoaded,
            )

        self.rag_factory = RAGChainFactory(
            retriever=self.datastore_manager.get_retriever()
        )
        self.rag_chain = self.rag_factory.get_rag_chain(self.llm)
        return self.rag_chain

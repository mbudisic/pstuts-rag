"""RAG (Retrieval Augmented Generation) module for PsTuts.

This module provides the core RAG functionality, including:
- RetrieverFactory: Creates and manages vector store retrievers
- RAGChainFactory: Builds RAG chains using LangChain components
"""

import json
import uuid
from operator import itemgetter
from typing import Dict, List, Any

from langchain_core.documents import Document
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from .datastore import initialize_vectorstore, transcripts_load
from .prompt_templates import RAG_PROMPT_TEMPLATES

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import AIMessage


class RetrieverFactory:
    """Factory class for creating and managing vector store retrievers.

    This class simplifies the process of creating, populating, and managing
    Qdrant vector stores for document retrieval.

    Attributes:
        embeddings: OpenAI embeddings model for document vectorization
        docs: List of documents stored in the vector store
        qdrant_client: Client for Qdrant vector database
        name: Unique identifier for this retriever instance
        vector_store: The Qdrant vector store instance
    """

    embeddings: OpenAIEmbeddings
    docs: List[Document]
    qdrant_client: QdrantClient
    name: str
    vector_store: QdrantVectorStore

    def __init__(
        self,
        embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
        qdrant_client: QdrantClient = QdrantClient(location=":memory:"),
        name: str = str(object=uuid.uuid4()),
    ) -> None:
        """Initialize the RetrieverFactory.

        Args:
            embeddings: OpenAI embeddings model to use
            qdrant_client: Qdrant client for vector database operations
            name: Unique identifier for this retriever instance
        """
        self.embeddings = embeddings
        self.name = name
        self.qdrant_client = qdrant_client
        self.vector_store = initialize_vectorstore(
            client=self.qdrant_client,
            collection_name=f"{self.name}_qdrant",
            embeddings=self.embeddings,
        )
        self.docs = []

    def add_docs(self, raw_docs: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store.

        Takes raw document data, converts it to Document objects,
        and adds them to the vector store.

        Args:
            raw_docs: List of raw document dictionaries
        """
        docs: List[Document] = transcripts_load(
            json_transcripts=raw_docs, embeddings=self.embeddings
        )
        self.docs.extend(docs)
        _ = self.vector_store.add_documents(documents=docs)

    def clear(self) -> bool:
        """Clear all documents from the vector store.

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self.docs = []
        return True if self.vector_store.delete() else False

    def get_retriever(self, n_context_docs: int = 2) -> VectorStoreRetriever:
        """Get a retriever for the vector store.

        Args:
            n_context_docs: Number of documents to retrieve for each query

        Returns:
            VectorStoreRetriever: The configured retriever
        """
        return self.vector_store.as_retriever(
            search_kwargs={"k": n_context_docs}
        )


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

        text_w_references = "\n".join(
            [answer.content, "**References**:", references]
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
            RAG_PROMPT_TEMPLATES
        )

    def get_rag_chain(
        self,
        llm: BaseLanguageModel = ChatOpenAI(
            model="gpt-4.1-mini", temperature=0
        ),
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

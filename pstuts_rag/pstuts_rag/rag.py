"""RAG (Retrieval Augmented Generation) module for PsTuts.

This module provides the core RAG functionality, including:
- RetrieverFactory: Creates and manages vector store retrievers
- RAGChainFactory: Builds RAG chains using LangChain components
"""

import json
import re

from operator import itemgetter
from typing import Any, Dict, List, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI

from .prompt_templates import RAG_PROMPT_TEMPLATES


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
            raise ValueError(
                f"No '**References:**' section found in input:\n{content}"
            )

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

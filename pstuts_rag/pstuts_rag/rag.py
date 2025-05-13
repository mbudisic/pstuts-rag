import json
import uuid
from operator import itemgetter
from typing import List, Optional, Union

from langchain.schema.output_parser import StrOutputParser
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

        self.embeddings = embeddings
        self.name = name
        self.qdrant_client = qdrant_client
        self.vector_store = initialize_vectorstore(
            client=self.qdrant_client,
            collection_name=f"{self.name}_qdrant",
            embeddings=self.embeddings,
        )
        self.docs = []

    def add_docs(self, raw_docs) -> None:

        docs: List[Document] = transcripts_load(
            json_transcripts=raw_docs, embeddings=self.embeddings
        )
        self.docs.extend(docs)
        _ = self.vector_store.add_documents(documents=docs)

    def clear(self) -> bool:
        self.docs = []
        return True if self.vector_store.delete() else False

    def get_retriever(self, n_context_docs: int = 2) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(
            search_kwargs={"k": n_context_docs}
        )


class RAGChainFactory:

    format_query = RunnableLambda(itemgetter("question"))
    retriever: VectorStoreRetriever
    add_context_to_query: Runnable
    prompt_template: Runnable
    answer_chain: Runnable
    llm: ChatOpenAI

    rag_chain: Runnable

    @staticmethod
    def compile_references(context: List[Document]) -> str:
        references = [
            {k: doc.metadata[k] for k in ("title", "source", "start", "stop")}
            for doc in context
        ]
        return json.dumps(references, indent=2)

    pack_references: RunnableLambda = RunnableLambda(
        lambda d: {
            **d["input"],
            "answer": d["answer"]
            + "\nReferences:\n"
            + RAGChainFactory.compile_references(
                context=d["input"]["context"]
            ),
        }
    )

    @staticmethod
    def pack_references2(msg_dict: Dict[str, Any]) -> AIMessage:

        answer: AIMessage = msg_dict["answer"]
        input = msg_dict["input"]

        references = RAGChainFactory.compile_references(
            context=input["context"]
        )

        text_w_references = f"{answer.content}\n**References**:\n{references}"

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

        self.answer_chain = self.prompt_template | llm
        self.rag_chain = (
            self.format_query
            | self.prepare_query
            | {"input": RunnablePassthrough(), "answer": self.answer_chain}
            | self.pack_references2
        )

        return self.rag_chain

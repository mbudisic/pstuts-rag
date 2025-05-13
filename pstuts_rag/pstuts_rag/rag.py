from typing import List, Optional, Union
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from .datastore import transcripts_load, initialize_vectorstore

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from pstuts_rag.loader import VideoTranscriptBulkLoader, VideoTranscriptLoader

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from langchain_core.retrievers import VectorStoreRetriever


import uuid


class RAGFactory:

    embeddings: OpenAIEmbeddings
    docs: List[Document]
    qdrantclient: QdrantClient
    name: str
    vectorstore: QdrantVectorStore
    

    def __init__(
        self,
        raw_docs: List[Documents],        
        embeddings: OpenAIEmbeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        qdrantclient: QdrantClient = QdrantClient(":memory:"),
        name: str = uuid.uuid4(),
    ) -> None:

        self.embeddings = embeddings
        self.name = name
        self.qdrantclient = qdrantclient
        self.vectorstore = initialize_vectorstore(
            client=self.qdrantclient,
            collection_name=f"{self.name}_qdrant",
            embeddings=self.embeddings,
        )
        self.docs = []


    def add_docs(raw_docs: List[Documents]):
        
        docs = transcripts_load(raw_docs, self.embeddings)
        self.docs.append( docs )
        _ = self.vectorstore.add_documents(documents=docs)
        
    def clear() -> bool:
        self.docs = []
        return self.vectorstore.delete()
                
        
    def get_retriever(self, n_context_docs=2) -> VectorStoreRetriever
        return self.vectorstore.as_retriever(search_kwargs={"k": n_context_docs})

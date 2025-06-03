import asyncio
import json
import glob
import aiofiles
from pathlib import Path
from typing import List, Dict, Iterator, Any, Callable, Optional, Self
import uuid
import logging

import chainlit as cl
from langchain_core.document_loaders import BaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_core.vectorstores import VectorStoreRetriever

from langchain_qdrant import QdrantVectorStore
from pstuts_rag.configuration import Configuration, ModelAPI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from pstuts_rag.utils import EmbeddingsAPISelector, flatten, batch


class DatastoreManager:
    """Factory class for creating and managing vector store retrievers.

    This class simplifies the process of creating, populating, and managing
    Qdrant vector stores for document retrieval.

    Attributes:
        embeddings: OpenAI embeddings model for document vectorization
        docs: List of documents stored in the vector store
        qdrant_client: Client for Qdrant vector database
        name: Unique identifier for this retriever instance
        vector_store: The Qdrant vector store instance
        loading_complete: AsyncIO event that's set when data loading completes
        _completion_callbacks: List of callbacks to call when loading completes
    """

    embeddings: Embeddings
    docs: List[Document]
    qdrant_client: QdrantClient
    name: str
    vector_store: QdrantVectorStore
    dimensions: int
    loading_complete: asyncio.Event
    _completion_callbacks: List[Callable]

    config: Optional[Configuration]

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        qdrant_client: QdrantClient = QdrantClient(location=":memory:"),
        name: str = str(object=uuid.uuid4()),
        config: Configuration = Configuration(),
    ) -> None:
        """Initialize the RetrieverFactory.

        Args:
            embeddings: OpenAI embeddings model to use
            qdrant_client: Qdrant client for vector database operations
            name: Unique identifier for this retriever instance
        """

        if embeddings is None:

            cls = EmbeddingsAPISelector.get(
                config.embedding_api, HuggingFaceEmbeddings
            )
            self.embeddings = cls(model=config.embedding_model)
        else:
            self.embeddings = embeddings

        self.name = name if name else config.eva_workflow_name
        self.qdrant_client = qdrant_client
        self.loading_complete = asyncio.Event()
        self._completion_callbacks = []

        # determine embedding dimension
        self.dimensions = len(self.embeddings.embed_query("test"))

        self.qdrant_client.recreate_collection(
            collection_name=self.name,
            vectors_config=VectorParams(
                size=self.dimensions, distance=Distance.COSINE
            ),
        )

        # wrapper around the client
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.name,
            embedding=self.embeddings,
        )

        self.docs = []

    async def from_json_globs(self, globs: List[str] | str) -> int:
        """
        Populate the vector database with processed video transcript documents,
        retrieved from JSON file paths.

        This method performs the complete pipeline:
            - loading JSON transcripts
            - semantic chunking with timestamp preservation
            -


        """
        logging.debug("Starting to load files.")
        files = globs_to_paths(globs)

        tasks = [load_json_file(f) for f in files]
        results = await asyncio.gather(*tasks)

        json_transcripts = list(flatten(results))
        logging.debug("Received %d JSON files.", len(json_transcripts))

        # perform chunking
        self.docs: List[Document] = await chunk_transcripts(
            json_transcripts=json_transcripts,
            semantic_chunker_embedding_model=self.embeddings,
        )

        count = await self.embed_chunks(self.docs)
        logging.debug("Uploaded %d records.", count)

        self.loading_complete.set()
        # Execute callbacks (both sync and async)
        for callback in self._completion_callbacks:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()

    async def embed_chunks(self, chunked_documents: List[Document]) -> int:
        """
        Populate the vector database with processed video transcript documents.

        This method performs the complete pipeline: semantic chunking of transcripts,
        embedding generation, and uploading to the Qdrant vector database. It processes
        the raw video data through semantic chunking, generates embeddings in batches
        for efficiency, and stores the results as vector points.

        Args:
            raw_docs (List[Dict[str, Any]]): List of raw video dictionaries containing
                                           transcript data, metadata, and other fields

        Returns:
            int: Number of document chunks successfully uploaded to the database

        Raises:
            Exception: If embedding generation or database upload fails
        """

        # perform embedding

        vector_batches = await asyncio.gather(
            *[
                self.embeddings.aembed_documents(
                    [c.page_content for c in chunk_batch]
                )
                for chunk_batch in batch(chunked_documents, 8)
            ]
        )
        vectors = []
        for vb in vector_batches:
            vectors.extend(vb)
        ids = list(range(len(vectors)))

        points = [
            PointStruct(
                id=id,
                vector=vector,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            for id, vector, doc in zip(ids, vectors, chunked_documents)
        ]

        # upload qdrant payload
        self.qdrant_client.upload_points(
            collection_name=self.name,
            points=points,
        )

        return len(points)

    def count_docs(self) -> int:
        """
        Get the current number of documents stored in the vector database.

        Returns:
            int: Number of points/documents in the Qdrant collection,
                 or 0 if collection doesn't exist or is empty

        Note:
            This method is safe to call even if the collection doesn't exist
        """
        try:
            count = self.qdrant_client.get_collection(self.name).points_count
            return count if count else 0
        except ValueError:
            return 0

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
            search_kwargs={"k": int(n_context_docs)}
        )

    def is_ready(self) -> bool:
        """Check if the datastore has finished loading data.

        Returns:
            bool: True if data loading is complete, False otherwise
        """
        return self.loading_complete.is_set()

    def add_completion_callback(self, callback: Callable) -> Self:
        """Add a callback to be called when data loading completes.

        Args:
            callback: Callable function to be called when data loading completes

        Note:
            If loading has already completed, the callback will be called immediately.
        """
        if self.loading_complete.is_set():
            # Loading already completed, execute callback immediately
            if asyncio.iscoroutinefunction(callback):
                # Need to schedule async callback
                asyncio.create_task(callback())
            else:
                callback()
        else:
            # Loading not complete, add to callbacks list
            self._completion_callbacks.append(callback)

        return self

    async def wait_for_loading(self, timeout: Optional[float] = None):
        """Wait for data loading to complete.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            bool: True if loading completed, False if timeout occurred
        """
        try:
            await asyncio.wait_for(
                self.loading_complete.wait(), timeout=timeout
            )
            return True
        except asyncio.TimeoutError:
            return False


async def load_json_file(filepath: Path):
    """
    Asynchronously load and parse a single JSON file containing video data.

    This function reads a JSON file asynchronously and processes it using
    load_json_string, setting the group field to the filename for identification.

    Args:
        filepath (str | Path): Path to the JSON file containing video transcript data.
                              Can be a string path or Path object.

    Returns:
        List[Dict]: List of video dictionaries with group field set to filename

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If file content is not valid JSON format
        PermissionError: If file cannot be read due to insufficient permissions
        UnicodeDecodeError: If file encoding is not UTF-8 compatible

    Note:
        Uses async file I/O for better performance when loading multiple files
    """

    async with aiofiles.open(filepath, mode="r", encoding="utf-8") as f:
        content = await f.read()
        payload = json.loads(content)
        for entry in payload:
            entry.update({"group": str(filepath)})
    return payload


def globs_to_paths(glob_list: List[str] | str) -> List[Path]:

    if isinstance(glob_list, str):
        glob_list = glob_list.split(":")

    logging.debug("Loading from %d globs:", len(glob_list))

    files: List[Path] = []
    for globstring in glob_list:
        logging.debug("Loading glob: %s", globstring)
        new_files = [Path(f) for f in glob.glob(globstring, recursive=True)]
        files.extend(filter(lambda f: f.exists(), new_files))

    logging.debug("Total files: %d", len(files))

    return files


def load_transcripts_whole(json_payload: List[Dict]) -> Iterator[Document]:
    """
    Lazy loader that yields Document objects with concatenated transcripts.

    Creates one Document per video with all transcript sentences joined by newlines.
    Metadata includes all video fields except 'transcripts' and 'qa'.
    The 'url' field is renamed to 'source' for LangChain compatibility.

    Yields:
        Document: LangChain Document with page_content as concatenated transcript
                    and metadata containing video information
    """

    for video in json_payload:
        metadata = dict(video)
        metadata.pop("transcripts", None)
        metadata.pop("qa", None)
        # Rename 'url' key to 'source' in metadata if it exists
        if "url" in metadata:
            metadata["source"] = metadata.pop("url")
        yield Document(
            page_content="\n".join(t["sent"] for t in video["transcripts"]),
            metadata=metadata,
        )


def load_transcripts_segments(
    json_payload: List[Dict],
) -> Iterator[Document]:
    """
    Lazy loader that yields individual Document objects for each transcript segment.

    Creates one Document per transcript segment with timing metadata.
    Each document contains a single transcript sentence with precise start/end times.
    The 'url' field is renamed to 'source' for LangChain compatibility.

    Yields:
        Document: LangChain Document with page_content as single transcript sentence
                    and metadata containing video info plus time_start and time_end
    """

    for video in json_payload:
        metadata = dict(video)
        transcripts = metadata.pop("transcripts", None)
        metadata.pop("qa", None)
        # Rename 'url' key to 'source' in metadata if it exists
        if "url" in metadata:
            metadata["source"] = metadata.pop("url")
        for transcript in transcripts:
            yield Document(
                page_content=transcript["sent"],
                metadata=metadata
                | {
                    "time_start": transcript["begin"],
                    "time_end": transcript["end"],
                },
            )


async def chunk_transcripts(
    json_transcripts: List[Dict[str, Any]],
    semantic_chunker_embedding_model: Embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    ),
) -> List[Document]:
    """
    Load and process video transcripts into semantically chunked documents.

    This function takes a list of transcript dictionaries, loads them as both full
    transcripts and individual chunks, then applies semantic chunking. It also
    enriches each semantic chunk with timestamp metadata from the original verbatim chunks.

    Args:
        json_transcripts: List of dictionaries containing video transcript data
        embeddings: OpenAI embeddings model to use for semantic chunking

    Returns:
        List of semantically chunked Document objects with enhanced metadata
    """

    docs_full_transcript: List[Document] = list(
        load_transcripts_whole(json_transcripts)
    )
    docs_chunks_verbatim: List[Document] = list(
        load_transcripts_segments(json_transcripts)
    )

    # semantically split the combined transcript
    text_splitter = SemanticChunker(semantic_chunker_embedding_model)
    docs_group = await asyncio.gather(
        *[
            text_splitter.atransform_documents(d)
            for d in batch(docs_full_transcript, size=2)
        ]
    )
    # Flatten the nested list of documents
    docs_chunks_semantic: List[Document] = []
    for group in docs_group:
        docs_chunks_semantic.extend(group)

    # Create a lookup dictionary for faster access
    video_id_to_chunks: Dict[int, List[Document]] = {}
    for chunk in docs_chunks_verbatim:
        video_id: int = chunk.metadata["video_id"]
        if video_id not in video_id_to_chunks:
            video_id_to_chunks[video_id] = []
        video_id_to_chunks[video_id].append(chunk)

    for chunk in docs_chunks_semantic:
        video_id = chunk.metadata["video_id"]
        # Only check chunks from the same video
        potential_subchunks = video_id_to_chunks.get(video_id, [])
        subchunks = [
            c
            for c in potential_subchunks
            if c.page_content in chunk.page_content
        ]

        times = [
            (t.metadata["time_start"], t.metadata["time_end"])
            for t in subchunks
        ]
        chunk.metadata["speech_start_stop_times"] = times

        if times:  # Avoid IndexError if times is empty
            chunk.metadata["start"], chunk.metadata["stop"] = (
                times[0][0],
                times[-1][-1],
            )
        else:
            chunk.metadata["start"], chunk.metadata["stop"] = None, None

    docs_chunks_semantic[0].metadata.keys()
    return docs_chunks_semantic

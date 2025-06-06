import asyncio
import atexit
from enum import Enum
import glob
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Self

import aiofiles
import chainlit as cl
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import QdrantVectorStore
from pathvalidate import sanitize_filename, sanitize_filepath
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from pstuts_rag.configuration import Configuration
from pstuts_rag.utils import batch, flatten, get_embeddings_api

# TODO: Write MCP server that ingests `mp4` folder


class QdrantClientSingleton:
    """
    Thread-safe singleton for QdrantClient. Ignores path changes after first initialization.
    Logs every invocation of get_client.
    """

    _instance = None
    _lock = threading.Lock()
    _config = None

    @classmethod
    def get_client(cls, path=None):
        import logging

        logging.info(
            f"QdrantClientSingleton.get_client called with path={path!r}"
        )
        from qdrant_client import QdrantClient

        with cls._lock:
            if cls._instance is None:
                if path is None:
                    cls._instance = QdrantClient(location=":memory:")
                    cls._config = ":memory:"
                else:
                    cls._instance = QdrantClient(path=path)
                    cls._config = path
            # Ignore any subsequent path changes, always return the first-initialized client
            return cls._instance


class LoadingState(Enum):
    NOT_STARTED = "not_started"
    LOADING = "loading"
    COMPLETE = "complete"


# TODO: accumulate transcripts of videos when loading, summarize each, then summarize summaries to get a description of the dataset for the prompt


class Datastore:
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
    collection_name: str
    name: str
    vector_store: QdrantVectorStore
    dimensions: int
    loading_complete: asyncio.Event
    loading_state: LoadingState = LoadingState.NOT_STARTED
    _completion_callbacks: List[Callable]
    config: Configuration
    reload: bool = True

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        qdrant_client: QdrantClient | None = None,
        name: str = "EVA_AI",
        config: Configuration = Configuration(),
    ) -> None:
        """Initialize the RetrieverFactory.

        Args:
            embeddings: OpenAI embeddings model to use
            qdrant_client: Qdrant client for vector database operations
            name: Unique identifier for this retriever instance
        """
        self.config = config
        if embeddings is None:
            cls = get_embeddings_api(config.embedding_api)
            self.embeddings = cls(model=config.embedding_model)
        else:
            self.embeddings = embeddings

        self.name = name if name else config.eva_workflow_name

        if qdrant_client is None:
            # Use the singleton for QdrantClient
            path = None
            if (
                config.db_persist
                and isinstance(config.db_persist, str)
                and len(config.db_persist) > 0
            ):
                qdrant_path = (
                    Path(sanitize_filepath(config.db_persist))
                    / f"{sanitize_filename(config.embedding_model)}_{sanitize_filename(config.transcript_glob)}"
                )
                logging.info(
                    "Persisting the datastore to: %s", str(qdrant_path)
                )
                qdrant_path.mkdir(parents=True, exist_ok=True)
                path = str(qdrant_path)
            qdrant_client = QdrantClientSingleton.get_client(path=path)

        self.qdrant_client = qdrant_client
        atexit.register(qdrant_client.close)

        self.loading_complete = asyncio.Event()
        self._completion_callbacks = []

        # determine embedding dimension
        self.dimensions = len(self.embeddings.embed_query("test"))
        self.collection_name = (
            sanitize_filename(config.transcript_glob) + "_transcripts"
        )
        # Try to create the collection, fall back to get_collection if it already exists
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimensions, distance=Distance.COSINE
                ),
            )
            logging.info(
                f"Collection {self.collection_name} created. Will reload."
            )
            self.reload = True
        except ValueError:

            count = self.count_docs()
            logging.info(
                f"Collection {self.collection_name} already exists (w/ {count} documents)."
            )

            self.reload = count == 0 or self.config.eva_reinitialize
            if self.reload:
                logging.warning("Reloading collection.")
            else:
                logging.info("Skipping reload.")

        # wrapper around the client
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
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

        if not (self.loading_state == LoadingState.NOT_STARTED):
            logging.info(
                "Cannot restart loading. Current state: %s", self.loading_state
            )
            return 0
        self.loading_state = LoadingState.LOADING

        doc_count = self.count_docs()
        if not (self.reload):
            self.loading_state = LoadingState.COMPLETE
            self.loading_complete.set()
            msg = f"Skipping initialization. Database holds {doc_count} documents."
            await cl.Message(content=msg).send()
            logging.warning(msg)
            return doc_count

        files = globs_to_paths(globs)
        logging.info(f"Starting to load {len(files)} files.")

        tasks = [load_json_file(f) for f in files]
        results = await asyncio.gather(*tasks)

        json_transcripts = list(flatten(results))
        logging.info(
            "Received %d JSON files. Chunking....", len(json_transcripts)
        )

        # perform chunking
        self.docs: List[Document] = await chunk_transcripts(
            json_transcripts=json_transcripts,
            semantic_chunker_embedding_model=self.embeddings,
        )
        logging.info("Embedding chunks...")
        count = await self.embed_chunks(self.docs)
        logging.info("Uploaded %d records.", count)

        self.loading_state = LoadingState.COMPLETE
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
        if self.count_docs() == len(points):
            logging.info("Qdrant database populated. Skipping upload")
        else:
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
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
            count = self.qdrant_client.get_collection(
                self.collection_name
            ).points_count
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
    semantic_chunker_embedding_model: Embeddings,
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

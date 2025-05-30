import asyncio
from pathlib import Path
from typing import List, Dict, Iterator, Any
import uuid


from langchain_core.document_loaders import BaseLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_core.vectorstores import VectorStoreRetriever

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import VectorParams, Distance, PointStruct


def batch(iterable: List[Any], size: int = 16) -> Iterator[List[Any]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


class VideoTranscriptBulkLoader(BaseLoader):
    """
    Loads video transcripts as bulk documents for document processing pipelines.

    Each video becomes a single document with all transcript sentences concatenated.
    Useful for semantic search across entire video content.

    Inherits from LangChain's BaseLoader for compatibility with document processing chains.

    Attributes:
        json_payload (List[Dict]): List of video dictionaries containing transcript data
    """

    def __init__(self, json_payload: List[Dict]):
        """
        Initialize the bulk loader with video transcript data.

        Args:
            json_payload (List[Dict]): List of video dictionaries, each containing:
                                     - transcripts: List of transcript segments
                                     - qa: Q&A data (optional)
                                     - url: Video URL
                                     - other metadata fields
        """

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy loader that yields Document objects with concatenated transcripts.

        Creates one Document per video with all transcript sentences joined by newlines.
        Metadata includes all video fields except 'transcripts' and 'qa'.
        The 'url' field is renamed to 'source' for LangChain compatibility.

        Yields:
            Document: LangChain Document with page_content as concatenated transcript
                     and metadata containing video information
        """

        for video in self.json_payload:
            metadata = dict(video)
            metadata.pop("transcripts", None)
            metadata.pop("qa", None)
            # Rename 'url' key to 'source' in metadata if it exists
            if "url" in metadata:
                metadata["source"] = metadata.pop("url")
            yield Document(
                page_content="\n".join(
                    t["sent"] for t in video["transcripts"]
                ),
                metadata=metadata,
            )


class VideoTranscriptChunkLoader(BaseLoader):
    """
    Loads video transcripts as individual chunk documents for fine-grained processing.

    Each transcript segment becomes a separate document with timing information.
    Useful for precise timestamp-based retrieval and time-sensitive queries.

    Inherits from LangChain's BaseLoader for compatibility with document processing chains.

    Attributes:
        json_payload (List[Dict]): List of video dictionaries containing transcript data
    """

    def __init__(self, json_payload: List[Dict]):
        """
        Initialize the chunk loader with video transcript data.

        Args:
            json_payload (List[Dict]): List of video dictionaries, each containing:
                                     - transcripts: List of transcript segments with timing
                                     - qa: Q&A data (optional)
                                     - url: Video URL
                                     - other metadata fields
        """

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy loader that yields individual Document objects for each transcript segment.

        Creates one Document per transcript segment with timing metadata.
        Each document contains a single transcript sentence with precise start/end times.
        The 'url' field is renamed to 'source' for LangChain compatibility.

        Yields:
            Document: LangChain Document with page_content as single transcript sentence
                     and metadata containing video info plus time_start and time_end
        """

        for video in self.json_payload:
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

    docs_full_transcript: List[Document] = VideoTranscriptBulkLoader(
        json_payload=json_transcripts
    ).load()
    docs_chunks_verbatim: List[Document] = VideoTranscriptChunkLoader(
        json_payload=json_transcripts
    ).load()

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

    # locate individual sections of the original transcript
    # with the semantic chunks
    def is_subchunk(a: Document, ofb: Document) -> bool:
        return (a.metadata["video_id"] == ofb.metadata["video_id"]) and (
            a.page_content in ofb.page_content
        )

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
    """

    embeddings: Embeddings
    docs: List[Document]
    qdrant_client: QdrantClient
    name: str
    vector_store: QdrantVectorStore
    dimensions: int

    def __init__(
        self,
        embeddings: Embeddings = OpenAIEmbeddings(
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

        # determine embedding dimension
        self.dimensions = len(embeddings.embed_query("test"))

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
            embedding=embeddings,
        )

        self.docs = []

    async def populate_database(self, raw_docs: List[Dict[str, Any]]) -> int:

        # perform chunking
        self.docs: List[Document] = await chunk_transcripts(
            json_transcripts=raw_docs,
            semantic_chunker_embedding_model=self.embeddings,
        )

        # perform embedding

        vector_batches = await asyncio.gather(
            *[
                self.embeddings.aembed_documents(
                    [c.page_content for c in chunk_batch]
                )
                for chunk_batch in batch(self.docs, 8)
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
            for id, vector, doc in zip(ids, vectors, self.docs)
        ]

        # upload qdrant payload
        self.qdrant_client.upload_points(
            collection_name=self.name,
            points=points,
        )

        return len(points)

    def count_docs(self) -> int:
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
            search_kwargs={"k": n_context_docs}
        )


def load_json_string(content: str, group: str):
    """
    Parse JSON string content and add group metadata to each video entry.

    Args:
        content (str): JSON string containing a list of video objects
        group (str): Group identifier to be added to each video entry

    Returns:
        List[Dict]: List of video dictionaries with added 'group' field

    Raises:
        json.JSONDecodeError: If content is not valid JSON
    """
    payload: List[Dict] = json.loads(content)
    [video.update({"group": group}) for video in payload]
    return payload


async def load_single_json(filepath):
    """
    Asynchronously load and parse a single JSON file containing video data.

    Args:
        filepath (str | Path): Path to the JSON file to load

    Returns:
        List[Dict]: List of video dictionaries with group field set to filename

    Raises:
        FileNotFoundError: If the specified file doesn't exist
        json.JSONDecodeError: If file content is not valid JSON
        PermissionError: If file cannot be read due to permissions
    """
    my_path = Path(filepath)

    async with aiofiles.open(my_path, mode="r", encoding="utf-8") as f:
        content = await f.read()
        payload = load_json_string(content, my_path.name)

    return payload


async def load_json_files(path_pattern: List[str]):
    """
    Asynchronously load and parse multiple JSON files matching given patterns.

    Uses glob patterns to find files and loads them concurrently for better performance.
    All results are flattened into a single list.

    Args:
        path_pattern (List[str]): List of glob patterns to match JSON files
                                 (supports recursive patterns with **)

    Returns:
        List[Dict]: Flattened list of all video dictionaries from matched files

    Raises:
        FileNotFoundError: If any matched file doesn't exist during loading
        json.JSONDecodeError: If any file content is not valid JSON
        PermissionError: If any file cannot be read due to permissions
    """
    files = []
    for f in path_pattern:
        (files.extend(glob.glob(f, recursive=True)))

    tasks = [load_single_json(f) for f in files]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]  # flatten

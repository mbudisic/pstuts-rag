from typing import List, Dict, Iterator, Any


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document

from .loader import VideoTranscriptBulkLoader, VideoTranscriptLoader

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


def transcripts_load(
    json_transcripts: List[Dict[str, Any]],
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
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
    docs_chunks_verbatim: List[Document] = VideoTranscriptLoader(
        json_payload=json_transcripts
    ).load()

    text_splitter = SemanticChunker(embeddings)

    docs_chunks_semantic: List[Document] = text_splitter.split_documents(
        docs_full_transcript
    )

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

    return docs_chunks_semantic


def initialize_vectorstore(
    client: QdrantClient, collection_name: str, embeddings: OpenAIEmbeddings
) -> QdrantVectorStore:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    return vector_store

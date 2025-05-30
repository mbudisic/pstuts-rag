import glob
import json
from langchain_core.document_loaders import BaseLoader
from typing import List, Dict, Iterator
from langchain_core.documents import Document

import aiofiles
import asyncio
from pathlib import Path


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

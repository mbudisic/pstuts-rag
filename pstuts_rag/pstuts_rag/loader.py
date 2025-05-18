import glob
import json
from langchain_core.document_loaders import BaseLoader
from typing import List, Dict, Iterator
from langchain_core.documents import Document

import aiofiles
import asyncio
from pathlib import Path


def load_json_string(content: str, group: str):
    payload: List[Dict] = json.loads(content)
    [video.update({"group": group}) for video in payload]
    return payload


async def load_single_json(filepath):
    my_path = Path(filepath)

    async with aiofiles.open(my_path, mode="r", encoding="utf-8") as f:
        content = await f.read()
        payload = load_json_string(content, my_path.name)

    return payload


async def load_json_files(path_pattern: List[str]):
    files = []
    for f in path_pattern:
        (files.extend(glob.glob(f, recursive=True)))

    tasks = [load_single_json(f) for f in files]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]  # flatten


class VideoTranscriptBulkLoader(BaseLoader):
    """Loads video transcripts as a bulk into documents"""

    def __init__(self, json_payload: List[Dict]):

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loader that returns an iterator"""

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
    """Loads video transcripts as individual chunks into documents"""

    def __init__(self, json_payload: List[Dict]):

        self.json_payload = json_payload

    def lazy_load(self) -> Iterator[Document]:
        """Lazy loader that returns an iterator"""

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

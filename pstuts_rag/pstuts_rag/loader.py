from langchain_core.document_loaders import BaseLoader
from typing import List, Dict, Iterator
from langchain_core.documents import Document


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
                page_content="\n".join(t["sent"] for t in video["transcripts"]),
                metadata=metadata,
            )


class VideoTranscriptLoader(BaseLoader):
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

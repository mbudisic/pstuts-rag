import operator
from typing import Annotated, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class PsTutsTeamState(BaseModel):
    """State management for PsTuts team workflow orchestration."""

    messages: Annotated[List[BaseMessage], operator.add] = Field(
        default_factory=list,
        description="Accumulated list of messages exchanged during team collaboration",
    )
    team_members: List[str] = Field(
        default_factory=list,
        description="List of active team member identifiers participating in the workflow",
    )
    next: str = Field(
        default="",
        description="Identifier of the next team member or process step to execute",
    )


class StateInput(BaseModel):
    """Input state for processing user queries and requests."""

    input_query: str = Field(
        description="The user's input query or topic to be processed by the system",
    )


class StateOutput(BaseModel):
    """Output state containing processed results and references."""

    output_content: str = Field(
        default="",
        description="The final generated content or report based on the input query",
    )
    video_references: List[Tuple[str, float]] = Field(
        default_factory=list,
        description="List of video references with starting timestamp",
    )
    url_references: List[str] = Field(
        default_factory=list,
        description="List of URL references related to the generated content",
    )


class RAGInput(BaseModel):
    """Input for Retrieval-Augmented Generation processing."""

    query: str = Field(
        description="The search query to be processed by the RAG system",
    )


class VideoSegment(BaseModel):
    """Summarized video segment"""

    path: str = Field(default="", description="Path to video file")
    summary: str = Field(default="", description="Summary of the transcript")
    start: Optional[float] = Field(description="Start time of the segment")
    stop: Optional[float] = Field(description="End time of the segment")


class RAGOutput(BaseModel):
    """Output from Retrieval-Augmented Generation processing."""

    query: str = Field(
        default="", description="The original query that was processed"
    )
    content: str = Field(
        default="",
        description="The generated content based on retrieved documents and query",
    )
    context: List[Dict[str, VideoSegment]] = Field(
        default_factory=list,
        description="Retrieved document context with relevance scores and metadata",
    )

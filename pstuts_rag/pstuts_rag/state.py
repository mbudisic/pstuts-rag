from dataclasses import dataclass, field
from langchain_core.messages import BaseMessage


import operator
from typing import Annotated, List, Tuple, TypedDict


class PsTutsTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str


@dataclass(kw_only=True)
class StateInput:
    input_query: str = field(default=None)  # Report topic


@dataclass(kw_only=True)
class StateOutput:
    output_content: str = field(default=None)  # Final report
    video_references: List[Tuple[str, float]] = field(default=None)
    url_references: List[str] = field(default=None)

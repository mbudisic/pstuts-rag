from langchain_core.messages import BaseMessage


import operator
from typing import Annotated, List, TypedDict


class PsTutsTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str

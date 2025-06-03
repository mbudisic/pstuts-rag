# nodes.py

from pstuts_rag.state import PsTutsTeamState
from langgraph.graph import StateGraph, MessagesState, START, END

from enum import Enum
from typing import Any, Callable, Dict


def eval_is_relevant(
    state: PsTutsTeamState,
):

    pass


def eval_is_complete(
    state: PsTutsTeamState,
):

    pass


def research(state: PsTutsTeamState):

    pass


def search_help(state: PsTutsTeamState):
    pass


def search_rag(state: PsTutsTeamState):
    pass


def join(state: PsTutsTeamState):
    pass


def write_answer(state: PsTutsTeamState):
    pass


class TutorialState(MessagesState):
    """State management for tutorial team workflow orchestration."""

    team_members: list[str]
    next: str
    video_references: set[Any]
    url_references: set[Any]


## MISSING: CONDITIONAL NODES


def route_is_relevant(state: PsTutsTeamState) -> bool:
    pass


def route_is_complete(state: PsTutsTeamState) -> bool:
    pass


graph_builder = StateGraph(PsTutsTeamState)

graph_builder.add_node(eval_is_relevant)
graph_builder.add_node(eval_is_complete)
graph_builder.add_node(research)
graph_builder.add_node(search_help)
graph_builder.add_node(search_rag)
graph_builder.add_node(join)
graph_builder.add_node(write_answer)

graph_builder.add_edge(START, eval_is_relevant.__name__)
graph_builder.add_conditional_edges(
    eval_is_relevant.__name__,
    route_is_relevant,
    {True: research.__name__, False: write_answer.__name__},
)

graph_builder.add_edge(research.__name__, search_help.__name__)
graph_builder.add_edge(research.__name__, search_rag.__name__)
graph_builder.add_edge(search_help.__name__, eval_is_complete.__name__)
graph_builder.add_edge(search_rag.__name__, eval_is_complete.__name__)
graph_builder.add_conditional_edges(
    eval_is_complete.__name__,
    route_is_complete,
    {True: write_answer.__name__, False: research.__name__},
)
graph_builder.add_edge(write_answer.__name__, END)

graph = graph_builder.compile()

# nodes.py

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pstuts_rag.prompts import NODE_PROMPTS
from pydantic import BaseModel, Field


from pstuts_rag.utils import ChatAPISelector
from pstuts_rag.configuration import Configuration

from enum import Enum
from typing import Any, Callable, Dict, Literal


class TutorialState(MessagesState):
    """State management for tutorial team workflow orchestration."""

    # next: str
    query: str
    video_references: set[Any]
    url_references: set[Any]


def research(state: TutorialState, config: RunnableConfig):

    # retrieve the LLM
    # configurable = Configuration.from_runnable_config(config)
    # cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    # llm = cls(model=configurable.llm_tool_model)

    # # format the prompt
    # prompt = NODE_PROMPTS["research"]

    # history = [
    #     msg.content
    #     for msg in state["messages"]
    #     if getattr(msg, "role", "") == "ai"
    # ]

    # prompt = prompt.format(history=history)

    pass


def search_help(state: TutorialState, config: RunnableConfig):
    pass


def search_rag(state: TutorialState, config: RunnableConfig):
    pass


def join(state: TutorialState, config: RunnableConfig):
    pass


def write_answer(state: TutorialState, config: RunnableConfig):
    pass


## MISSING: CONDITIONAL NODES


class YesNoDecision(BaseModel):
    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")


def route_is_relevant(
    state: TutorialState, config: RunnableConfig
) -> Command[Literal["research", "write_answer"]]:

    # retrieve the LLM
    configurable = Configuration.from_runnable_config(config)
    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model).with_structured_output(
        YesNoDecision
    )

    # format the prompt
    prompt = NODE_PROMPTS["relevance"].format(query=state["query"])

    relevance = llm.invoke([HumanMessage(content=prompt)])
    where = "research" if relevance.decision == "yes" else "write_answer"
    answer = f"Query is {'not' if relevance.decision == 'no' else ''} relevant to Photoshop."
    return Command(
        update={"messages": {"role": "ai", "content": answer}},
        goto=where,
    )


def route_is_complete(
    state: TutorialState, config: RunnableConfig
) -> Literal["yes", "no"]:
    if True:
        return "yes"
    else:
        return "no"


graph_builder = StateGraph(TutorialState)

# graph_builder.add_node(route_is_relevant)
# graph_builder.add_node(route_is_complete, defer=True)
graph_builder.add_node(research)
graph_builder.add_node(search_help)
graph_builder.add_node(search_rag)
graph_builder.add_node(join)
graph_builder.add_node(write_answer)

# graph_builder.add_conditional_edges(
#     START,
#     route_is_relevant,
#     {"yes": research.__name__, "no": write_answer.__name__},
# )
graph_builder.add_node(route_is_relevant)
graph_builder.add_edge(START, route_is_relevant.__name__)
graph_builder.add_edge(research.__name__, search_help.__name__)
graph_builder.add_edge(research.__name__, search_rag.__name__)
graph_builder.add_edge(search_help.__name__, join.__name__)
graph_builder.add_edge(search_rag.__name__, join.__name__)
graph_builder.add_conditional_edges(
    join.__name__,
    route_is_complete,
    {"no": research.__name__, "yes": write_answer.__name__},
)
graph_builder.add_edge(write_answer.__name__, END)


graph = graph_builder.compile()

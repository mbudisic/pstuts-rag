# nodes.py
from enum import Enum
from typing import Annotated, Any, Callable, Dict, Literal

import asyncio
import logging
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from numpy import add
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilyExtract

from pydantic import BaseModel, Field, HttpUrl


from pstuts_rag.utils import ChatAPISelector
from pstuts_rag.configuration import Configuration
from pstuts_rag.datastore import DatastoreManager
from pstuts_rag.prompts import NODE_PROMPTS
from pstuts_rag.rag_for_transcripts import create_transcript_rag_chain


class TutorialState(MessagesState):
    """State management for tutorial team workflow orchestration."""

    # next: str
    query: str
    video_references: Annotated[list[Document], operator.add]
    url_references: Annotated[list[Dict], operator.add]
    loop_count: int


datastore = DatastoreManager()
datastore.add_completion_callback(lambda: logging.warning("Loading complete."))


def research(state: TutorialState, config: RunnableConfig):

    configurable = Configuration.from_runnable_config(config)
    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model, temperature=0)

    history = [
        msg.content
        for msg in state["messages"]
        if getattr(msg, "role", "") == "ai"
    ]

    prompt = NODE_PROMPTS["research"].format(
        history=history, query=state["query"]
    )

    search_query = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [search_query],
        "loop_count": state.get("loop_count", 0) + 1,
    }


async def search_help(
    state: TutorialState, config: RunnableConfig | None = None
):

    configurable = (
        Configuration()
        if not config
        else Configuration.from_runnable_config(config)
    )

    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model, temperature=0)
    prompt = NODE_PROMPTS["search_summary"]

    adobe_help_search = TavilySearchResults(
        max_results=2,
        include_domains=["helpx.adobe.com"],
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        response_format="content_and_artifact",  # Always returns artifacts
    )
    query = state["messages"][-1].content
    results = await adobe_help_search.ainvoke(query)

    urls = list(r["url"] for r in results)
    tool = TavilyExtract(
        extract_depth="basic",
        include_images=False,
    )

    results = await tool.ainvoke({"urls": urls})

    if "results" in results:
        all_text = list(r["raw_content"] for r in results["results"])
    else:
        all_text = []

    prompt = prompt.format(
        query=query,
        text="\n***\n".join(all_text),
    )

    url_summary = await llm.ainvoke([HumanMessage(content=prompt)])

    return {"messages": [url_summary], "url_references": results["results"]}


async def search_rag(state: TutorialState, config: RunnableConfig):

    chain = create_transcript_rag_chain(datastore, config)

    response = await chain.ainvoke({"question": state["messages"][-1].content})

    return {
        "messages": [response],
        "video_references": response.additional_kwargs["context"],
    }


def join(state: TutorialState, config: RunnableConfig):
    pass


def write_answer(state: TutorialState, config: RunnableConfig):
    pass


## MISSING: CONDITIONAL NODES


class YesNoDecision(BaseModel):
    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")


class URLReference(BaseModel):
    summary: str
    url: HttpUrl


def route_is_relevant(
    state: TutorialState, config: RunnableConfig
) -> Command[Literal["research", "write_answer"]]:

    # retrieve the LLM
    configurable = Configuration.from_runnable_config(config)
    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model).with_structured_output(
        YesNoDecision
    )

    human_messages = [
        msg.content
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]

    if len(human_messages) > 0:
        query = human_messages[-1]
    else:
        query = state["query"]

    # format the prompt
    prompt = NODE_PROMPTS["relevance"].format(query=query)

    relevance = llm.invoke([HumanMessage(content=prompt)])
    where = "research" if relevance.decision == "yes" else "write_answer"
    answer = (
        f"Query is {'not' if relevance.decision == 'no' else ''} "
        "relevant to Photoshop."
    )
    return Command(
        update={"messages": [AIMessage(content=answer)], "query": query},
        goto=where,
    )


class IsComplete(BaseModel):
    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")
    new_query: str = Field(description="Query for additional research.")


def route_is_complete(
    state: TutorialState, config: RunnableConfig
) -> Command[Literal["research", "write_answer"]]:

    # retrieve the LLM
    configurable = Configuration.from_runnable_config(config)

    if state["loop_count"] >= int(configurable.max_research_loops):
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="Research loop count is too large. Do your best with what you have."
                    )
                ]
            },
            goto="write_answer",
        )

    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model).with_structured_output(
        YesNoDecision
    )

    ai_messages = list(
        msg.content for msg in state["messages"] if isinstance(msg, AIMessage)
    )

    # format the prompt
    prompt = NODE_PROMPTS["completeness"].format(
        query=state["query"], responses="\n\n".join(ai_messages)
    )

    completeness = llm.invoke([HumanMessage(content=prompt)])
    where = "write_answer" if "yes" in completeness.decision else "research"

    # Convert YesNoDecision to AIMessage
    decision_message = AIMessage(
        content=f"Research completeness: {completeness.decision}"
    )

    return Command(
        update={"messages": [decision_message]},
        goto=where,
    )


def write_answer(state: TutorialState, config: RunnableConfig):

    # retrieve the LLM
    configurable = Configuration.from_runnable_config(config)
    cls = ChatAPISelector.get(configurable.llm_api, ChatOpenAI)
    llm = cls(model=configurable.llm_tool_model)

    ai_messages = list(
        msg.content for msg in state["messages"] if isinstance(msg, AIMessage)
    )

    # format the prompt
    prompt = NODE_PROMPTS["final_answer"].format(
        query=state["query"], responses="\n\n".join(ai_messages)
    )

    final_answer = llm.invoke([HumanMessage(content=prompt)])

    return {"messages": [final_answer]}


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
graph_builder.add_node(route_is_complete, defer=True)

graph_builder.add_edge(START, route_is_relevant.__name__)
graph_builder.add_edge(research.__name__, search_help.__name__)
graph_builder.add_edge(research.__name__, search_rag.__name__)
graph_builder.add_edge(search_help.__name__, route_is_complete.__name__)
graph_builder.add_edge(search_rag.__name__, route_is_complete.__name__)

graph_builder.add_edge(write_answer.__name__, END)


graph = graph_builder.compile()
asyncio.run(datastore.from_json_globs(Configuration().transcript_glob))

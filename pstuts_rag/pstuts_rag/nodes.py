# nodes.py
from enum import Enum
from typing import Annotated, Any, Callable, Dict, Literal, Tuple, Union
import functools
import asyncio
import logging
import operator
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from numpy import add
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilyExtract

from pydantic import BaseModel, Field, HttpUrl


from pstuts_rag.utils import ChatAPISelector
from pstuts_rag.configuration import Configuration
from pstuts_rag.datastore import DatastoreManager
from pstuts_rag.prompts import NODE_PROMPTS
from pstuts_rag.rag_for_transcripts import create_transcript_rag_chain


class YesNoAsk(Enum):
    """Enumeration for user permission states regarding search operations.

    Attributes:
        YES: Permission granted for search operations
        NO: Permission denied for search operations
        ASK: Permission should be requested from user interactively
    """

    YES = "yes"
    NO = "no"
    ASK = "ask"

    @classmethod
    def from_string(cls, value: str, default=None) -> "YesNoAsk":
        """Parse enum from string with optional default fallback.

        Args:
            value: String value to parse (case-insensitive)
            default: Default enum value if parsing fails (defaults to ASK)

        Returns:
            YesNoAsk: Parsed enum value or default if parsing fails
        """
        if default is None:
            default = cls.ASK

        try:
            return cls(value.lower().strip())
        except ValueError:
            return default


class TutorialState(MessagesState):
    """State management for tutorial team workflow orchestration."""

    # next: str
    query: str
    video_references: Annotated[list[Document], operator.add]
    url_references: Annotated[list[Dict], operator.add]
    loop_count: int
    search_permission: YesNoAsk


class QueryMessage(AIMessage):
    """A message class representing a research query, retaining all attributes from any message type."""

    @classmethod
    def from_message(cls, message: BaseMessage) -> "QueryMessage":
        """Create a QueryMessage from any BaseMessage type, retaining all attributes.

        Args:
            message: Any BaseMessage instance (HumanMessage, AIMessage, SystemMessage, etc.)

        Returns:
            QueryMessage: New QueryMessage instance with all attributes from the source message
        """
        # Get all attributes from the source message
        message_dict = message.model_dump()

        # Create new QueryMessage with all attributes preserved
        return cls(**message_dict)

    def __init__(self, content: Union[str, BaseMessage] = "", **kwargs):
        """Initialize QueryMessage from content string or any BaseMessage type.

        Args:
            content: Either a string content or a BaseMessage instance
            **kwargs: Additional keyword arguments for the message
        """
        if isinstance(content, BaseMessage):
            # If content is a BaseMessage, convert it
            source_dict = content.model_dump()
            # Merge kwargs into source_dict, with kwargs taking precedence
            source_dict.update(kwargs)
            super().__init__(**source_dict)
        else:
            # Normal string content initialization
            super().__init__(content=content, **kwargs)


def research(state: TutorialState, config: RunnableConfig):
    """Generate a research query based on conversation history and current query.

    Args:
        state: Current TutorialState containing messages and query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with new message and incremented loop count
    """

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
        "messages": [QueryMessage(search_query)],
        "loop_count": state.get("loop_count", 0) + 1,
    }


from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver


async def search_help(
    state: TutorialState, config: RunnableConfig | None = None
) -> Command[Literal["search_help", "route_is_complete"]]:
    """Search Adobe Help documentation for relevant information.

    Args:
        state: Current TutorialState containing the search query
        config: Optional RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with search results message and URL references
    """

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
    queries = [
        msg.content
        for msg in state["messages"]
        if isinstance(msg, QueryMessage)
    ]
    query = queries[-1]

    decision = state["search_permission"]
    if decision == YesNoAsk.ASK:

        response = interrupt(
            (
                f"Do you allow Internet search for query '{query}'?"
                "Answer 'yes' will perform the search, any other answer will skip it."
            )
        )

        logging.info(f"Permission response '{response}'")
        decision = YesNoAsk.YES if "yes" in response.strip() else YesNoAsk.NO
        return Command(
            update={"search_permission": decision}, goto=search_help.__name__
        )

    response = {
        "messages": [],
        "url_references": [],
        "search_permission": YesNoAsk.from_string(
            configurable.search_permission
        ),
    }

    if decision == YesNoAsk.YES:
        longform = f"Query '{query}' is permitted."
    else:
        longform = f"Query '{query}' is NOT permitted."

    response["messages"].append({"role": "human", "content": longform})

    if decision == YesNoAsk.YES:

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
        response["messages"].append(url_summary)
        response["url_references"].extend(results["results"])

    return Command(update=response, goto=route_is_complete.__name__)


async def search_rag(
    state: TutorialState, config: RunnableConfig, datastore: DatastoreManager
):
    """Search tutorial transcripts using RAG (Retrieval-Augmented Generation).

    Args:
        state: Current TutorialState containing the search query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with RAG response and video references
    """

    chain = create_transcript_rag_chain(datastore, config)

    response = await chain.ainvoke({"question": state["messages"][-1].content})

    return {
        "messages": [response],
        "video_references": response.additional_kwargs["context"],
    }


def write_answer(state: TutorialState, config: RunnableConfig):
    """Write a preliminary answer (placeholder function).

    Args:
        state: Current TutorialState with research data
        config: RunnableConfig for accessing configuration parameters

    Returns:
        None: Currently a placeholder function
    """
    pass


## MISSING: CONDITIONAL NODES


class YesNoDecision(BaseModel):
    """Model for yes/no decision responses from LLM.

    Attributes:
        decision: The yes or no decision as a literal string
    """

    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")


class URLReference(BaseModel):
    """Model for URL reference with summary.

    Attributes:
        summary: Text summary of the URL content
        url: The HTTP URL being referenced
    """

    summary: str
    url: HttpUrl


def route_is_relevant(
    state: TutorialState, config: RunnableConfig
) -> Command[Literal["research", "write_answer"]]:
    """Route based on whether the query is relevant to Photoshop tutorials.

    Args:
        state: Current TutorialState containing the user query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        Command: Navigation command to either 'research' or 'write_answer'
    """

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
    """Model for research completion decision.

    Attributes:
        decision: Whether research is complete (yes/no)
        new_query: Query string for additional research if needed
    """

    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")
    new_query: str = Field(description="Query for additional research.")


def route_is_complete(
    state: TutorialState, config: RunnableConfig
) -> Command[Literal["research", "write_answer"]]:
    """Route based on whether research is complete or more is needed.

    Args:
        state: Current TutorialState with research progress
        config: RunnableConfig for accessing configuration parameters

    Returns:
        Command: Navigation command to either 'research' or 'write_answer'
    """

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
    """Generate the final answer based on all research collected.

    Args:
        state: Current TutorialState with all research data
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with the final answer message
    """

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


# Lazy initialization: compiled graph is cached
_compiled_graph = None
_datastore = None
_checkpointer = MemorySaver()


def init_state(_: TutorialState, config: RunnableConfig):
    """Initialize the default state for the tutorial workflow.

    Args:
        _: TutorialState (unused, for function signature compatibility)
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Default state dictionary with search permission setting
    """
    configurable = Configuration.from_runnable_config(config)

    default_state = {
        "search_permission": YesNoAsk.from_string(
            configurable.search_permission
        )
    }

    return default_state


def initialize(
    datastore: DatastoreManager | None = None,
    configuration: Configuration = Configuration(),
) -> Tuple[DatastoreManager, StateGraph]:
    """Initialize and configure the LangGraph StateGraph with all nodes and edges.

    Creates a complete workflow graph with research, search, and routing nodes.
    Sets up datastore for RAG operations and configures node connections.

    Args:
        datastore: Optional pre-configured DatastoreManager instance
        configuration: Configuration object for system settings

    Returns:
        tuple: (DatastoreManager instance, configured StateGraph builder)
    """
    if datastore is None:
        datastore = DatastoreManager(
            config=Configuration()
        ).add_completion_callback(lambda: "Datastore loading completed.")

    graph_builder = StateGraph(TutorialState)

    # graph_builder.add_node(route_is_relevant)
    # graph_builder.add_node(route_is_complete, defer=True)
    graph_builder.add_node(init_state)
    graph_builder.add_node(research)
    graph_builder.add_node(search_help)
    graph_builder.add_node(
        "search_rag", functools.partial(search_rag, datastore=datastore)
    )
    graph_builder.add_node(write_answer)

    # graph_builder.add_conditional_edges(
    #     START,
    #     route_is_relevant,
    #     {"yes": research.__name__, "no": write_answer.__name__},
    # )
    graph_builder.add_node(route_is_relevant)
    graph_builder.add_node(route_is_complete, defer=True)
    graph_builder.add_edge(START, init_state.__name__)

    graph_builder.add_edge(init_state.__name__, route_is_relevant.__name__)
    graph_builder.add_edge(research.__name__, search_help.__name__)
    graph_builder.add_edge(research.__name__, search_rag.__name__)
    # graph_builder.add_edge(search_help.__name__, route_is_complete.__name__)
    graph_builder.add_edge(search_rag.__name__, route_is_complete.__name__)

    graph_builder.add_edge(write_answer.__name__, END)

    return datastore, graph_builder


async def graph(config: RunnableConfig = None):
    """Graph factory function for LangGraph Studio compatibility.

    This function provides lazy initialization of the graph and datastore,
    allowing the module to be imported without triggering compilation.
    LangGraph Studio requires this function to take exactly one
    RunnableConfig argument.

    Args:
        config: RunnableConfig (required by LangGraph Studio, but can be None)

    Returns:
        Compiled LangGraph instance
    """
    global _compiled_graph
    global _datastore
    global _checkpointer

    # Initialize datastore using asyncio.to_thread to avoid blocking
    initialize_datastore: bool = _datastore is None
    if initialize_datastore:
        _datastore = await asyncio.to_thread(
            lambda: DatastoreManager(
                config=Configuration()
            ).add_completion_callback(lambda: "Datastore loading completed.")
        )

    # Initialize and compile graph synchronously (blocking as intended)
    if _compiled_graph is None:
        _datastore, graph_builder = initialize(_datastore)
        _compiled_graph = graph_builder.compile(checkpointer=_checkpointer)

    # Start datastore population as background task (non-blocking)
    if initialize_datastore:
        asyncio.create_task(
            _datastore.from_json_globs(Configuration().transcript_glob)
        )

    return _compiled_graph

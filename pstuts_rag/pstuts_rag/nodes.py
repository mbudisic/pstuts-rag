# nodes.py
import asyncio
import functools
import logging
import operator
import os
from enum import Enum
from typing import Annotated, Dict, Literal, Tuple

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_tavily import TavilyExtract
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field, HttpUrl

from pstuts_rag.configuration import Configuration
from pstuts_rag.datastore import Datastore
from pstuts_rag.prompts import NODE_PROMPTS
from pstuts_rag.rag_for_transcripts import create_transcript_rag_chain
from pstuts_rag.utils import get_chat_api, get_title_streaming


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

    next: str  # Routing variable for next node
    query: str
    video_references: Annotated[list[Document], operator.add]
    url_references: Annotated[list[Dict], operator.add]
    loop_count: int
    search_permission: YesNoAsk
    search_query: Annotated[list[str], operator.add]


# Define your FinalAnswer message type as a subclass of BaseMessage
class FinalAnswer(AIMessage):
    """Custom message type for final streaming responses"""

    type: str = "FinalAnswer"

    def __init__(self, content: str = "", **kwargs):
        super().__init__(content=content, **kwargs)

    @classmethod
    def from_base_message(cls, base_msg: BaseMessage) -> "FinalAnswer":
        """Create FinalAnswer from any BaseMessage, copying all relevant fields"""
        return cls(
            content=base_msg.content,
            additional_kwargs=getattr(base_msg, "additional_kwargs", {}),
            response_metadata=getattr(base_msg, "response_metadata", {}),
            name=getattr(base_msg, "name", None),
            id=getattr(base_msg, "id", None),
        )


class QueryMessage(AIMessage):
    """Custom message type for final streaming responses"""

    type: str = "QueryMessage"

    def __init__(self, content: str = "", **kwargs):
        super().__init__(content=content, **kwargs)

    @classmethod
    def from_base_message(cls, base_msg: BaseMessage) -> "QueryMessage":
        """Create QueryMessage from any BaseMessage, copying all relevant fields"""
        return cls(
            content=base_msg.content,
            additional_kwargs=getattr(base_msg, "additional_kwargs", {}),
            response_metadata=getattr(base_msg, "response_metadata", {}),
            name=getattr(base_msg, "name", None),
            id=getattr(base_msg, "id", None),
        )


# TODO More robust generation of queries - multiquery launch
def research(state: TutorialState, config: RunnableConfig):
    """Generate a research query based on conversation history and current query.

    Args:
        state: Current TutorialState containing messages and query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with new message and incremented loop count
    """

    configurable = Configuration.from_runnable_config(config)
    cls = get_chat_api(configurable.llm_api)
    llm = cls(model=configurable.llm_tool_model, temperature=0)

    history = [
        msg.content
        for msg in state["messages"]
        if not (isinstance(msg, HumanMessage))
    ]

    prompt = NODE_PROMPTS["research"].format(
        history=history,
        query=state["query"],
        previous_queries="\n\n".join(state["search_query"][0:-1]),
    )

    search_query = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [QueryMessage.from_base_message(search_query)],
        "loop_count": state.get("loop_count", 0) + 1,
        "search_query": [search_query.content],
    }


async def search_help(state: TutorialState, config: RunnableConfig):
    """Search Adobe Help documentation for relevant information.

    Args:
        state: Current TutorialState containing the search query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with search results message and URL references
    """

    configurable = await asyncio.to_thread(
        Configuration.from_runnable_config, config
    )
    logging.info("search_help: loaded config")
    cls = get_chat_api(configurable.llm_api)
    llm = cls(model=configurable.llm_tool_model, temperature=0)
    prompt = NODE_PROMPTS["search_summary"]

    logging.info("search_help: configured llm")

    adobe_help_search = TavilySearchResults(
        max_results=2,
        include_domains=["helpx.adobe.com"],
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        response_format="content_and_artifact",  # Always returns artifacts
    )
    logging.info("search_help: configured tavily")

    query = state["search_query"][-1]

    decision = state["search_permission"]
    if decision == YesNoAsk.ASK:
        logging.info("search_help: asking permission")

        response = interrupt(
            {
                "message": "Do you allow Internet search for this query?",
                "query": query,
            }
        )

        logging.info(f"Permission response '{response}'")
        decision = (
            YesNoAsk.YES
            if any(
                affirmative in response.strip().lower()
                for affirmative in [
                    "yes",
                    "true",
                    "ok",
                    "okay",
                    "1",
                    "y",
                    "sure",
                    "fine",
                    "alright",
                ]
            )
            else YesNoAsk.NO
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
    logging.info("search_help: %s", longform)

    response["messages"].append({"role": "human", "content": longform})

    if decision == YesNoAsk.YES:
        logging.info("search_help: searching")

        results = await adobe_help_search.ainvoke(query)

        logging.info("search_help: results")

        urls = list(r["url"] for r in results)
        tool = TavilyExtract(
            extract_depth="advanced",
            include_images=False,
        )
        logging.info("search_help: extract text")

        results = await tool.ainvoke({"urls": urls})

        if "results" in results:
            all_text = list(r["raw_content"] for r in results["results"])

            for r in results["results"]:
                r["title"] = get_title_streaming(r["url"])

        else:
            all_text = []

        prompt = prompt.format(
            query=query,
            text="\n***\n".join(all_text),
        )
        logging.info("search_help: text extracted. summarizing.")

        url_summary = await llm.ainvoke([HumanMessage(content=prompt)])
        response["messages"].append(url_summary)
        response["url_references"].extend(results["results"])
        logging.info("search_help: summary complete.")

    return response


# use advanced ensemble retrieval BM25 w/ dense vector
async def search_rag(
    state: TutorialState, config: RunnableConfig, datastore: Datastore
):
    """Search tutorial transcripts using RAG (Retrieval-Augmented Generation).

    Args:
        state: Current TutorialState containing the search query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: Updated state with RAG response and video references
    """

    configurable = await asyncio.to_thread(
        Configuration.from_runnable_config, config
    )
    chain = create_transcript_rag_chain(datastore, configurable)
    query = state["search_query"][-1]

    response = await chain.ainvoke({"question": query})

    return {
        "messages": [response],
        "video_references": response.additional_kwargs["context"],
    }


## MISSING: CONDITIONAL NODES


class YesNoDecision(BaseModel):
    """Model for yes/no decision responses from LLM.

    Attributes:
        decision: The yes or no decision as a literal string
    """

    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")

    @classmethod
    def from_string(cls, value: str) -> "YesNoDecision":
        """Parse a string and return a YesNoDecision instance, mapping common affirmatives to 'yes', others to 'no'."""
        affirmatives = {
            "yes",
            "y",
            "true",
            "ok",
            "okay",
            "sure",
            "1",
            "fine",
            "alright",
        }
        if value.strip().lower() in affirmatives:
            return cls(decision="yes")
        return cls(decision="no")


class URLReference(BaseModel):
    """Model for URL reference with summary.

    Attributes:
        summary: Text summary of the URL content
        url: The HTTP URL being referenced
    """

    summary: str
    url: HttpUrl


def route_is_relevant(
    state: TutorialState,
    config: RunnableConfig,
):
    """Route based on whether the query is relevant to Photoshop tutorials.

    Args:
        state: Current TutorialState containing the user query
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: State update with 'next' set to the next node name
    """
    configurable = Configuration.from_runnable_config(config)
    cls = get_chat_api(configurable.llm_api)
    logging.info("LLM SELECTED: %s", cls)

    llm = cls(
        model=configurable.llm_tool_model, temperature=0
    ).with_structured_output(YesNoDecision)

    human_messages = [
        msg.content
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]

    if len(human_messages) > 0:
        query = human_messages[-1]
    else:
        query = state["query"]

    prompt = NODE_PROMPTS["relevance"].format(query=query)
    relevance = llm.invoke([HumanMessage(content=prompt)])
    next_node = "research" if relevance.decision == "yes" else "write_answer"
    answer = (
        f"Query is {'not' if relevance.decision == 'no' else ''} "
        "relevant to Photoshop."
    )
    return {
        "messages": [AIMessage(content=answer)],
        "query": query,
        "next": next_node,
    }


class IsComplete(BaseModel):
    """Model for research completion decision.

    Attributes:
        decision: Whether research is complete (yes/no)
        new_query: Query string for additional research if needed
    """

    decision: Literal["yes", "no"] = Field(description="Yes or no decision.")
    new_query: str = Field(description="Query for additional research.")


def route_is_complete(state: TutorialState, config: RunnableConfig):
    """Route based on whether research is complete or more is needed.

    Args:
        state: Current TutorialState with research progress
        config: RunnableConfig for accessing configuration parameters

    Returns:
        dict: State update with 'next' set to the next node name
    """
    configurable = Configuration.from_runnable_config(config)

    if state["loop_count"] >= int(configurable.max_research_loops):
        return {
            "messages": [
                AIMessage(
                    content="Research loop count is too large. Do your best with what you have."
                )
            ],
            "next": "write_answer",
        }

    cls = get_chat_api(configurable.llm_api)
    logging.info("LLM SELECTED: %s", cls)

    llm = cls(model=configurable.llm_tool_model).with_structured_output(
        YesNoDecision
    )

    ai_messages = list(
        msg.content for msg in state["messages"] if isinstance(msg, AIMessage)
    )

    prompt = NODE_PROMPTS["completeness"].format(
        query=state["query"],
        responses="\n\n".join(str(m) for m in ai_messages),
    )

    completeness = llm.invoke([HumanMessage(content=prompt)])
    next_node = (
        "write_answer" if "yes" in completeness.decision else "research"
    )
    decision_message = AIMessage(
        content=f"Research completeness: {completeness.decision}"
    )
    return {"messages": [decision_message], "next": next_node}


# TODO: Produce Markdown artifact with embedded references
# TODO: Consider reviewing the document and interspersing the links based on their descriptions where most needed


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
    cls = get_chat_api(configurable.llm_api)

    llm = cls(model=configurable.llm_tool_model)

    ai_messages = list(
        msg.content for msg in state["messages"] if isinstance(msg, AIMessage)
    )

    # format the prompt
    prompt = NODE_PROMPTS["final_answer"].format(
        query=state["query"], responses="\n\n".join(ai_messages)
    )

    final_answer = FinalAnswer.from_base_message(
        llm.invoke([HumanMessage(content=prompt)])
    )

    return {"messages": [final_answer]}


# Lazy initialization: compiled graph is cached
_compiled_graph = None
_datastore = None
_checkpointer = MemorySaver()


def init_state(state: TutorialState, config: RunnableConfig):
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

    human = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if len(human) > 0 and len(state["query"]) == 0:
        default_state["query"] = human[0].content

    return default_state


def initialize(
    datastore: Datastore | None = None,
    configuration: Configuration = Configuration(),
) -> Tuple[Datastore, StateGraph]:
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
        datastore = Datastore(config=Configuration()).add_completion_callback(
            lambda: "Datastore loading completed."
        )

    graph_builder = StateGraph(TutorialState)

    graph_builder.add_node(init_state)
    graph_builder.add_node(research)
    graph_builder.add_node(search_help)
    graph_builder.add_node(
        "search_rag", functools.partial(search_rag, datastore=datastore)
    )
    graph_builder.add_node(write_answer)
    graph_builder.add_node(route_is_relevant)
    graph_builder.add_node(route_is_complete, defer=True)
    graph_builder.add_edge(START, init_state.__name__)
    graph_builder.add_edge(init_state.__name__, route_is_relevant.__name__)
    graph_builder.add_edge(research.__name__, search_help.__name__)
    graph_builder.add_edge(research.__name__, search_rag.__name__)
    graph_builder.add_edge(search_help.__name__, route_is_complete.__name__)
    graph_builder.add_edge(search_rag.__name__, route_is_complete.__name__)
    graph_builder.add_edge(write_answer.__name__, END)

    # Conditional edges for routing based on 'next' in state
    graph_builder.add_conditional_edges(
        route_is_relevant.__name__,
        lambda state: state["next"],
        {"research": research.__name__, "write_answer": write_answer.__name__},
    )
    graph_builder.add_conditional_edges(
        route_is_complete.__name__,
        lambda state: state["next"],
        {"research": research.__name__, "write_answer": write_answer.__name__},
    )

    return datastore, graph_builder


from dotenv import load_dotenv

load_dotenv()
logging.info("OLLAMA_HOST: %s", os.getenv("OLLAMA_HOST"))


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
            lambda: Datastore(config=Configuration()).add_completion_callback(
                lambda: "Datastore loading completed."
            )
        )

    # Initialize and compile graph synchronously (blocking as intended)
    if _compiled_graph is None:
        _datastore, graph_builder = initialize(_datastore)
        _compiled_graph = graph_builder.compile(checkpointer=_checkpointer)

    # Start datastore population as background task (non-blocking)
    if initialize_datastore:

        async def load_datastore():
            configurable = await asyncio.to_thread(Configuration)
            await asyncio.to_thread(
                _datastore.from_json_globs, configurable.transcript_glob
            )

        asyncio.create_task(load_datastore())

    return _compiled_graph

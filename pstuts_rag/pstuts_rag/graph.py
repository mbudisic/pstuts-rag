from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent import AgentExecutor
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from pstuts_rag.prompts import SUPERVISOR_SYSTEM, TAVILY_SYSTEM
from pstuts_rag.state import PsTutsTeamState
from pstuts_rag.datastore import Datastore
from pstuts_rag.configuration import Configuration

import asyncio
import functools
import logging
from typing import Callable, Dict, Tuple, Optional, Union

from langchain_huggingface import HuggingFaceEmbeddings
from pstuts_rag.utils import ChatAPISelector

from app import (
    ADOBEHELP,
    VIDEOARCHIVE,
    ApplicationState,
    app_state,
    enter_chain,
)

from pstuts_rag.rag_for_transcripts import create_transcript_rag_chain


def search_agent(state: PsTutsTeamState, chain: Runnable) -> Dict:
    """Extract search query from state and execute it using the provided chain.

    Attempts to extract a query from the state's input field or the last message,
    then executes the search using the provided chain.

    Args:
        state: PsTutsTeamState containing input or messages
        chain: Runnable chain to execute the search query

    Returns:
        Dict: Dictionary with 'output' key containing search results or error message
    """
    question = state.get("input", None)
    if not question and state.get("messages", []):
        last_message = state["messages"][-1]
        question = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

    logging.info("RAG question: %s", question)

    if not question:
        return {"output": "No query found in input or messages"}

    result = chain.invoke({"question": question})
    return {"output": result}


def agent_node(
    state: PsTutsTeamState,
    agent: Runnable,
    name: str,
    output_field: str = "output",
):
    """Execute an agent and format its output as an AIMessage.

    This function serves as a wrapper for agent execution within the graph,
    ensuring consistent message formatting and state management.

    Args:
        state: Current PsTutsTeamState for the conversation
        agent: Runnable agent to execute
        name: Name to assign to the AI message
        output_field: Field name to extract from agent result (default: "output")

    Returns:
        Dict: Dictionary with 'messages' key containing formatted AIMessage
    """
    # Initialize team_members if it's not already in the state
    if "team_members" not in state:
        state["team_members"] = []
    result = agent.invoke(state)

    # Check if result[output_field] is already an AIMessage
    if isinstance(result[output_field], AIMessage):
        # Extract just the content string from the AIMessage
        content = result[output_field].content
    else:
        content = result[output_field]

    return {"messages": [AIMessage(content=content, name=name)]}


def create_rag_node(rag_chain: Runnable, name: str = "VideoSearch"):
    """Create a RAG node for the agent graph.

    Args:
        rag_chain: RAG chain used in searching
        name: The name of the node

    Returns:
        tuple: (node, search_function)
    """

    # Use agent_node to create the node
    rag_node = functools.partial(
        agent_node,
        agent=RunnableLambda(functools.partial(search_agent, chain=rag_chain)),
        name=name,
    )

    return rag_node


def create_agent(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str,
):
    """Create a function-calling agent with the specified tools and prompt.

    Builds a LangChain agent that can use function calling with provided tools,
    configured with a custom system prompt and agent executor.

    Args:
        llm: Language model to power the agent
        tools: List of tools available to the agent
        system_prompt: System prompt to configure agent behavior

    Returns:
        AgentExecutor: Configured agent executor ready for use
    """
    system_prompt += pstuts_rag.prompts.AGENT_SYSTEM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def create_tavily_node(
    name: str = "AdobeHelp", config: Configuration = Configuration()
) -> Callable:
    """Initialize tool, agent, and node for Tavily search of helpx.adobe.com.

    This function sets up a search agent that can query Adobe Photoshop help topics
    using the Tavily search engine, specifically targeting helpx.adobe.com.

    Args:
        name: The name to assign to the agent node (defaults to "AdobeHelp")
        config: Configuration object for LLM and other settings

    Returns:
        Callable: A node function that can be added to a LangGraph
    """

    cls = ChatAPISelector.get(config.llm_api, ChatOpenAI)
    llm = cls(model=config.llm_tool_model)

    adobe_help_search = TavilySearchResults(
        max_results=5, include_domains=["helpx.adobe.com"]
    )
    adobe_help_agent = create_agent(
        llm=llm, tools=[adobe_help_search], system_prompt=TAVILY_SYSTEM
    )
    adobe_help_node = functools.partial(
        agent_node, agent=adobe_help_agent, name=name
    )

    return adobe_help_node


def create_team_supervisor(
    system_prompt,
    members,
    config: Configuration = Configuration(),
):
    """Create an LLM-based router to coordinate team member selection.

    Builds a supervisor agent that routes requests to appropriate team members
    or finishes the conversation based on the current context.

    Args:
        system_prompt: System prompt to guide routing decisions
        members: List of team member names available for routing
        config: Configuration object for LLM settings

    Returns:
        Runnable: Configured routing chain with function calling
    """
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }

    cls = ChatAPISelector.get(config.llm_api, ChatOpenAI)
    llm = cls(model=config.llm_tool_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", pstuts_rag.prompts.SUPERVISOR_SYSTEM),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_tools(
            tools=[function_def],
            tool_choice={"type": "function", "function": {"name": "route"}},
        )
        | JsonOutputFunctionsParser()
    )


def initialize_datastore(callback: Optional[Callable] = None):
    """Initialize and start loading the datastore in the background.

    Creates a DatastoreManager instance and begins asynchronous loading
    of transcript data from configured JSON glob patterns.

    Args:
        callback: Optional callback function to execute when loading completes

    Returns:
        DatastoreManager: Configured datastore instance with loading in progress
    """
    datastore = Datastore()
    if callback:
        datastore.add_completion_callback(callback)
    asyncio.create_task(
        datastore.from_json_globs(Configuration().transcript_glob)
    )

    return datastore


async def build_the_graph(
    datastore: Datastore, config: Configuration = Configuration()
):
    """
    Builds the agent graph for routing user queries.

    Creates the necessary nodes (Adobe help, RAG search, supervisor), defines their
    connections, and compiles the graph into a runnable chain.

    Args:
        current_state: Current application state with required components
    """

    adobe_help_node = create_tavily_node(name=ADOBEHELP, config=config)

    rag_node = create_transcript_rag_chain(datastore, config=config)

    supervisor_agent = create_team_supervisor(
        SUPERVISOR_SYSTEM, [VIDEOARCHIVE, ADOBEHELP], config=config
    )

    ai_graph = StateGraph(PsTutsTeamState, config_schema=Configuration)

    ai_graph.add_node(VIDEOARCHIVE, rag_node)
    ai_graph.add_node(ADOBEHELP, adobe_help_node)
    ai_graph.add_node("supervisor", supervisor_agent)

    ai_graph.add_edge(VIDEOARCHIVE, "supervisor")
    ai_graph.add_edge(ADOBEHELP, "supervisor")

    ai_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            VIDEOARCHIVE: VIDEOARCHIVE,
            ADOBEHELP: ADOBEHELP,
            "FINISH": END,
        },
    )

    ai_graph.set_entry_point("supervisor")

    return enter_chain | ai_graph.compile(), ai_graph


# Note: Cannot run build_the_graph() here as it requires current_state parameter
db = initialize_datastore(lambda _: logging.info("Database initialized"))
graph = asyncio.run(build_the_graph(db))

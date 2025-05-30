from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.agent import AgentExecutor
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda
from pstuts_rag.prompts import TAVILY_SYSTEM
from pstuts_rag.state import PsTutsTeamState


import logging
from typing import Callable, Dict, Tuple


def search_agent(state: PsTutsTeamState, chain: Runnable) -> Dict:
    """Extracts the search query from the current message chain in state."""
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
    """agent_node calls the invoke function of the agent Runnable"""
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

    return rag_node, lambda q: {"result": rag_chain.invoke({"question": q})}


def create_agent(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str,
):
    """Create a function-calling agent and add it to the graph."""
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
    llm: BaseChatModel, name: str = "AdobeHelp"
) -> Tuple[Callable, AgentExecutor, TavilySearchResults]:
    """Initialize tool, agent, and node for Tavily search of helpx.adobe.com.

    This function sets up a search agent that can query Adobe Photoshop help topics
    using the Tavily search engine, specifically targeting helpx.adobe.com.

    Args:
        llm: The language model to power the agent.
        name: The name to assign to the agent node. Defaults to "AdobeHelp".

    Returns:
        Tuple containing:
            - A callable node function that can be added to a graph
            - The configured agent executor
            - The Tavily search tool instance
    """

    adobe_help_search = TavilySearchResults(
        max_results=5, include_domains=["helpx.adobe.com"]
    )
    adobe_help_agent = create_agent(
        llm=llm, tools=[adobe_help_search], system_prompt=TAVILY_SYSTEM
    )
    adobe_help_node = functools.partial(
        agent_node, agent=adobe_help_agent, name=name
    )

    return adobe_help_node, adobe_help_agent, adobe_help_search


def create_team_supervisor(llm: BaseChatModel, system_prompt, members):
    """An LLM-based router."""
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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", pstuts_rag.prompts.SUPERVISOR_SYSTEM),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

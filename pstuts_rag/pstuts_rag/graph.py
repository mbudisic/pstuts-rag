from langchain.agents.agent import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableLambda
from pstuts_rag.agents import agent_node, create_agent
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

from typing import Callable, Tuple
from langchain_community.tools.tavily_search import TavilySearchResults
from .agents import create_agent, agent_node
from langchain_core.language_models.chat_models import BaseChatModel
import functools
from langchain.agents.agent import AgentExecutor
from .prompt_templates import TAVILY_SYSTEM


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

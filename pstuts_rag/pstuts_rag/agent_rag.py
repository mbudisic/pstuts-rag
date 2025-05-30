from typing import Dict
from langchain_core.runnables import RunnableLambda, Runnable
from .state import PsTutsTeamState
from .agents import agent_node
import functools
import logging


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

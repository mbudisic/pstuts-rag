from pstuts_rag.prompt_templates import AGENT_SYSTEM, RAG_PROMPT_TEMPLATES
from .rag import RAGChainFactory
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from .agents import agent_node
from langchain_core.language_models.chat_models import BaseChatModel
import functools


def create_rag_node(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    name: str = "VideoSearch",
):
    """Create a RAG node for the agent graph.

    Args:
        llm: The language model to use
        retriever: The retriever to use for RAG
        name: The name of the node

    Returns:
        tuple: (node, search_function)
    """
    rag_factory = RAGChainFactory(retriever=retriever)
    rag_chain = rag_factory.get_rag_chain(llm=llm)

    def search_transcripts(query: str) -> str:
        """Search through video transcripts to find relevant information about Photoshop.

        Args:
            query: The search query about Photoshop features or techniques

        Returns:
            str: Relevant information from the video transcripts
        """
        result = rag_chain.invoke(
            {"question": query, "input": query, "query": query}
        )
        return result.content

    # Create a simple agent that just does the search
    def search_agent(state):
        # Extract query from input or messages
        query = state.get("input", None)
        if not query and state.get("messages", []):
            last_message = state["messages"][-1]
            query = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )

        if not query:
            return {"output": "No query found in input or messages"}

        result = search_transcripts(query)
        return {"output": result}

    # Use agent_node to create the node
    rag_node = functools.partial(
        agent_node, agent=RunnableLambda(search_agent), name=name
    )

    return rag_node, search_transcripts

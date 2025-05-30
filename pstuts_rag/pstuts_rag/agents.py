from typing import Any, Callable, Optional, Union
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from langgraph.graph import END, StateGraph
import pstuts_rag.prompts
from pstuts_rag.state import PsTutsTeamState


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

import os
import logging
import pandas as pd
from typing import Any, Dict, List
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.prebuilt import create_react_agent
from langgraph.graph.graph import CompiledGraph
from langchain.agents.agent import AgentExecutor
from .tools import create_pdf_retriever_tool
from .schema import DomainGraphState
from .gpts import GPT_MODEL

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


def create_retriever_agent(vectordb: Chroma, system_prompt: str) -> CompiledGraph:
    """
    Create a specialized agent that retrieves PDF text using a Chroma retriever.

    Args:
        vectordb (Chroma): The vector database used by the PDF retriever tool.
        system_prompt (str): A system prompt used to modify the agent's state.

    Returns:
        CompiledGraph: A compiled React agent configured for PDF retrieval.
    """
    retriever_tool = create_pdf_retriever_tool(vectordb=vectordb)
    return create_react_agent(
        GPT_MODEL, tools=[retriever_tool], state_modifier=system_prompt
    )


def create_dataframe_agent(df: pd.DataFrame, system_prompt: str) -> AgentExecutor:
    """
    Create a specialized agent to analyze a pandas DataFrame.

    NOTE: 'create_pandas_dataframe_agent' allows arbitrary code execution; use with caution.

    Args:
        df (pd.DataFrame): The pandas DataFrame to be analyzed.
        system_prompt (str): A system prompt to modify the agent's state.

    Returns:
        AgentExecutor: An agent executor configured for DataFrame analysis.
    """
    agent = create_pandas_dataframe_agent(
        GPT_MODEL,
        df,
        agent_type="tool-calling",
        verbose=False,  # Set to True for debugging.
        allow_dangerous_code=True,  # Warning: this may execute dangerous code.
        prefix=system_prompt,
        max_execution_time=30,
    )
    agent.handle_parsing_errors = True
    return agent


def _call_agent(
    agent: Any,
    messages: List[Dict[str, str]],
    message_name: str,
    state: DomainGraphState,
    state_key: str,
    is_retriever: bool = False,
) -> dict:
    """
    Invoke an agent with provided messages, update the domain graph state, and return the updated state for a key.

    Args:
        agent (Any): The agent (or compiled graph) to be invoked.
        messages (List[Dict[str, str]]): A list of message dictionaries (with keys like 'role' and 'content').
        message_name (str): The name associated with the human message to append.
        state (DomainGraphState): The current state of the domain graph.
        state_key (str): The key in the state dictionary that will be updated with the agent's output.
        is_retriever (bool, optional): If True, the agent is invoked in retriever mode; defaults to False.

    Returns:
        dict: The updated state value for the specified key.
    """
    try:
        logger.info("Invoking agent for %s", state_key)
        if is_retriever:
            result = agent.invoke({"messages": messages})
            output = result["messages"][-1].content if result.get("messages") else ""
        else:
            result = agent.invoke(messages)
            output = result.get("output", "")
    except Exception as e:
        logger.error("Error in invoking agent for %s: %s", state_key, e)
        output = ""  # Default to empty if agent call fails

    updated_value = state.get(state_key, "")
    updated_value += "\n" + output

    messages_list = state.get("messages", [])
    messages_list.append(HumanMessage(content=output, name=message_name))
    return {
        "messages": messages_list,
        state_key: updated_value,
    }


def call_dataframe_agent(
    agent: AgentExecutor,
    messages: List[Dict[str, str]],
    message_name: str,
    state: DomainGraphState,
    state_key: str,
) -> dict:
    """
    Invoke a DataFrame agent to process messages and update the domain state.

    Args:
        agent (AgentExecutor): The agent executor for DataFrame analysis.
        messages (List[Dict[str, str]]): A list of message dictionaries.
        message_name (str): The name assigned to the human message.
        state (DomainGraphState): The current domain graph state.
        state_key (str): The key in the state to update with the agent's output.

    Returns:
        dict: The updated state.
    """
    return _call_agent(
        agent, messages, message_name, state, state_key, is_retriever=False
    )


def call_retriever_agent(
    agent: CompiledGraph,
    messages: List[Dict[str, str]],
    message_name: str,
    state: DomainGraphState,
    state_key: str,
) -> dict:
    """
    Invoke a PDF retriever agent to process messages and update the domain state.

    Args:
        agent (CompiledGraph): The compiled react agent for PDF retrieval.
        messages (List[Dict[str, str]]): A list of message dictionaries.
        message_name (str): The name assigned to the human message.
        state (DomainGraphState): The current domain graph state.
        state_key (str): The key in the state to update with the agent's output.

    Returns:
        dict: The updated state.
    """
    return _call_agent(
        agent, messages, message_name, state, state_key, is_retriever=True
    )

import os
import sys

# Support direct execution of this script by adding project root to paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from Src.Agents.chat_bot_node import chat_bot
from langchain_core.documents import Document

@pytest.fixture
def mock_agent_state():
    return {
        "query": "Who is Gunjan Tailor?",
        "messages": [],
        "document": [(Document(page_content="Gunjan Tailor is a software engineer."), 0.9)],
        "query_evaluator": "search"
    }

@patch("Src.Utils.llm_utils.get_llm")
def test_chat_bot_search_intent(mock_get_llm, mock_agent_state):
    # Setup mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Gunjan Tailor is a software engineer based on the context."
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    # Execute
    result = chat_bot(mock_agent_state)

    # Verify
    assert "response" in result
    assert "messages" in result
    assert result["response"] == mock_response.content
    assert isinstance(result["messages"][0], AIMessage)
    assert mock_get_llm.called

@patch("Src.Utils.llm_utils.get_llm")
def test_chat_bot_greeting_intent(mock_get_llm, mock_agent_state):
    # Setup state for greeting
    mock_agent_state["query_evaluator"] = "greeting"
    mock_agent_state["query"] = "Hello!"
    mock_agent_state["document"] = []

    # Setup mock LLM
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hello! How can I help you today?"
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    # Execute
    result = chat_bot(mock_agent_state)

    # Verify
    assert result["response"] == mock_response.content
    assert "Respond warmly" in str(mock_llm.method_calls[0]) # Check if prompt contained greeting instructions (indirectly via chain invoke)

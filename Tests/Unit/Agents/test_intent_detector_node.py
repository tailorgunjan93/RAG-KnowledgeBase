import os
import sys

# Support direct execution of this script by adding project root to paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest
from unittest.mock import MagicMock, patch
from Src.Agents.intent_detector_node import intent_detector, IntentClassification
from langchain_core.messages import HumanMessage

@pytest.fixture
def mock_agent_state():
    return {"query": "Hello AI!", "messages": []}

def test_intent_detector_heuristic_greeting(mock_agent_state):
    # Execute
    result = intent_detector(mock_agent_state)

    # Verify
    assert result["query_evaluator"] == "greeting"
    assert "messages" in result
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Hello AI!"

@patch("Src.Utils.llm_utils.get_llm")
def test_intent_detector_llm_search(mock_get_llm, mock_agent_state):
    # Setup state for search query (doesn't match heuristic)
    mock_agent_state["query"] = "Tell me about RAG architecture."

    # Setup mock LLM with structured output
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    
    # Mock result from structured LLM
    mock_classification = IntentClassification(intent="search")
    mock_structured.invoke.return_value = mock_classification
    
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    # Execute
    result = intent_detector(mock_agent_state)

    # Verify
    assert result["query_evaluator"] == "search"
    assert mock_llm.with_structured_output.called
    assert mock_structured.invoke.called

@patch("Src.Utils.llm_utils.get_llm")
def test_intent_detector_exception_fallback(mock_get_llm, mock_agent_state):
    # Setup state for search query
    mock_agent_state["query"] = "Complex query that causes error."
    
    # Force exception
    mock_get_llm.side_effect = Exception("LLM failure")

    # Execute
    result = intent_detector(mock_agent_state)

    # Verify fallback to search
    assert result["query_evaluator"] == "search"

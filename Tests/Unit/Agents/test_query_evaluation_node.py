import pytest
from unittest.mock import MagicMock, patch
from Src.Agents.query_evaluation_node import query_evaluation, RouteDecision
from langchain_core.documents import Document

@pytest.fixture
def mock_agent_state():
    return {"query": "Who is the PM of India?", "document": []}

@patch("Src.Agents.query_evaluation_node.QueryGrader")
def test_query_evaluation_relevant_docs(mock_query_grader, mock_agent_state):
    # Setup state with a document
    doc = Document(page_content="Gunjan Tailor is an engineer.")
    mock_agent_state["document"] = [(doc, 0.9)]
    
    # Mock QueryGrader returns 'yes'
    mock_query_grader.return_value = "yes"

    # Execute
    result = query_evaluation(mock_agent_state)

    # Verify
    assert result["query_evaluator"] == "chat_bot"
    mock_query_grader.assert_called_once()

@patch("Src.Utils.llm_utils.get_llm")
@patch("Src.Agents.query_evaluation_node.QueryGrader")
def test_query_evaluation_no_relevant_docs_route_web(mock_query_grader, mock_get_llm, mock_agent_state):
    # Setup state with irrelevant document
    doc = Document(page_content="Random text.")
    mock_agent_state["document"] = [(doc, 0.1)]
    
    # Mock QueryGrader returns 'no'
    mock_query_grader.return_value = "no"

    # Setup mock LLM for routing
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_decision = RouteDecision(route="web_search")
    mock_structured.invoke.return_value = mock_decision
    
    mock_llm.with_structured_output.return_value = mock_structured
    mock_get_llm.return_value = mock_llm

    # Execute
    result = query_evaluation(mock_agent_state)

    # Verify
    assert result["query_evaluator"] == "web_search"
    assert mock_llm.with_structured_output.called

@patch("Src.Utils.llm_utils.get_llm")
def test_query_evaluation_exception_fallback(mock_get_llm, mock_agent_state):
    # Force exception in LLM
    mock_get_llm.side_effect = Exception("Routing error")

    # Execute
    result = query_evaluation(mock_agent_state)

    # Verify fallback to chat_bot
    assert result["query_evaluator"] == "chat_bot"

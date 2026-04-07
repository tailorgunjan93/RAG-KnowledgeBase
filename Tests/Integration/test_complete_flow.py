import pytest
from unittest.mock import MagicMock, patch
from Src.Agents.Graph_builder import app
from langchain_core.messages import HumanMessage

@pytest.fixture
def mock_config():
    return {"configurable": {"thread_id": "test_thread"}}

@patch("Src.Utils.llm_utils.get_llm")
@patch("Src.Agents.query_evaluation_node.QueryGrader")
@patch("Src.Agents.retriever_node.search_neo4j_graph")
@patch("Src.Agents.retriever_node.search_dynamic_faiss_index_with_score")
@patch("Src.Agents.result_evaluator_node.result_evaluater")
def test_complete_flow_greeting(mock_res_eval, mock_faiss, mock_neo4j, mock_q_grader, mock_get_llm, mock_config):
    # Setup mock LLM for chat_bot
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Hi there! How can I help you?"
    mock_llm.invoke.return_value = mock_response
    mock_get_llm.return_value = mock_llm

    # Input state
    inputs = {"query": "Hello", "messages": []}

    # Execute graph
    final_state = app.invoke(inputs, config=mock_config)

    # Verify flow
    assert final_state["query_evaluator"] == "greeting"
    assert "Hi there" in final_state["response"]
    # For greetings, retriever should NOT be called (based on intent_decider in Graph_builder)
    assert not mock_neo4j.called
    assert not mock_faiss.called

@patch("Src.Utils.llm_utils.get_llm")
@patch("Src.Agents.query_evaluation_node.QueryGrader")
@patch("Src.Agents.retriever_node.search_neo4j_graph")
@patch("Src.Agents.retriever_node.search_dynamic_faiss_index_with_score")
@patch("Src.Agents.result_evaluator_node.result_evaluater")
def test_complete_flow_search(mock_res_eval, mock_faiss, mock_neo4j, mock_q_grader, mock_get_llm, mock_config):
    # Setup mocks
    mock_neo4j.return_value = []
    mock_faiss.return_value = []
    mock_q_grader.return_value = "no" # No relevant docs
    mock_res_eval.return_value = "yes" # Hallucination check passes
    
    # Mock LLM for routing and chat
    mock_llm = MagicMock()
    
    # Mock for query_evaluation (RouteDecision)
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = MagicMock(route="chat_bot")
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Mock for chat_bot response
    mock_response = MagicMock()
    mock_response.content = "I couldn't find specific info, but I can help generally."
    mock_llm.invoke.return_value = mock_response
    
    mock_get_llm.return_value = mock_llm

    # Input state
    inputs = {"query": "What is the meaning of life?", "messages": []}

    # Execute graph
    final_state = app.invoke(inputs, config=mock_config)

    # Verify flow
    assert final_state["query_evaluator"] == "chat_bot"
    assert mock_neo4j.called
    assert mock_faiss.called
    assert mock_res_eval.called

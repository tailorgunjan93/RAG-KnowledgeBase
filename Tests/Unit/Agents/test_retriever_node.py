import pytest
from unittest.mock import patch, MagicMock
from Src.Agents.retriever_node import retriever
from langchain_core.documents import Document

@pytest.fixture
def mock_agent_state():
    return {"query": "What is the capital of France?"}

@patch("Src.Agents.retriever_node.search_neo4j_graph")
@patch("Src.Agents.retriever_node.search_dynamic_faiss_index_with_score")
def test_retriever_combines_results(mock_faiss, mock_neo4j, mock_agent_state):
    # Setup mocks
    doc_neo4j = Document(page_content="Paris is the capital of France.")
    doc_faiss = Document(page_content="France's capital is Paris.")
    
    mock_neo4j.return_value = [(doc_neo4j, 0.0)]
    mock_faiss.return_value = [(doc_faiss, 0.5)]

    # Execute
    result = retriever(mock_agent_state)

    # Verify
    assert "document" in result
    assert len(result["document"]) == 2
    assert result["document"][0][0].page_content == "Paris is the capital of France."
    assert result["document"][1][0].page_content == "France's capital is Paris."
    
    mock_neo4j.assert_called_once_with(query=mock_agent_state["query"])
    mock_faiss.assert_called_once_with(query=mock_agent_state["query"])

@patch("Src.Agents.retriever_node.search_neo4j_graph")
@patch("Src.Agents.retriever_node.search_dynamic_faiss_index_with_score")
def test_retriever_empty_results(mock_faiss, mock_neo4j, mock_agent_state):
    # Setup mocks for empty results
    mock_neo4j.return_value = []
    mock_faiss.return_value = []

    # Execute
    result = retriever(mock_agent_state)

    # Verify
    assert result["document"] == []

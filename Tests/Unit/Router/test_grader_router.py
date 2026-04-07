import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from Src.Router.GraderRouter import GradeRouter
from langchain_core.documents import Document

client = TestClient(GradeRouter)

@pytest.fixture
def mock_docs():
    return [(Document(page_content="Gunjan Tailor is an AI engineer."), 0.9)]

@patch("Src.Router.GraderRouter.search_service")
@patch("Src.Router.GraderRouter.QueryGrader")
def test_grade_checker_success(mock_query_grader, mock_search_service, mock_docs):
    # Setup mocks
    mock_search_service.retrieve.return_value = mock_docs
    mock_query_grader.return_value = "yes"

    # Execute
    response = client.get("/Grade", params={"query": "Who is Gunjan?"})

    # Verify
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["grade"] == "yes"
    assert data[0]["document"] == "Gunjan Tailor is an AI engineer."
    assert data[0]["score"] == 0.9
    
    mock_search_service.retrieve.assert_called_once_with("Who is Gunjan?")
    mock_query_grader.assert_called_once()

@patch("Src.Router.GraderRouter.search_service")
def test_grade_checker_search_failure(mock_search_service):
    # Setup mock to raise error
    mock_search_service.retrieve.side_effect = Exception("Search failed")

    # Execute
    response = client.get("/Grade", params={"query": "Who is Gunjan?"})

    # Verify
    assert response.status_code == 503
    assert "Vector search failed" in response.json()["detail"]

@patch("Src.Router.GraderRouter.search_service")
@patch("Src.Router.GraderRouter.QueryGrader")
def test_grade_checker_grader_failure(mock_query_grader, mock_search_service, mock_docs):
    # Setup mocks
    mock_search_service.retrieve.return_value = mock_docs
    mock_query_grader.side_effect = Exception("Grader failed")

    # Execute
    response = client.get("/Grade", params={"query": "Who is Gunjan?"})

    # Verify
    assert response.status_code == 503
    assert "Grader failed" in response.json()["detail"]

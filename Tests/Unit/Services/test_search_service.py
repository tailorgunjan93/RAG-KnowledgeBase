import pytest
from unittest.mock import MagicMock, patch
from Src.Services.SearchService import SearchService, VectorStoreFactory
from langchain_core.documents import Document

@pytest.fixture
def mock_graph_store():
    mock = MagicMock()
    mock.query.return_value = "Graph answer"
    return mock

@pytest.fixture
def mock_vector_factory():
    mock = MagicMock()
    mock.search_best.return_value = [(Document(page_content="Vector answer"), 0.5)]
    return mock

@pytest.fixture
def mock_llm_provider():
    return MagicMock()

def test_search_service_retrieves_combined(mock_graph_store, mock_vector_factory, mock_llm_provider):
    service = SearchService(mock_vector_factory, mock_graph_store, mock_llm_provider)
    
    results = service.retrieve("test query")
    
    assert len(results) == 2
    assert "[GRAPH KNOWLEDGE]" in results[0][0].page_content
    assert "Vector answer" in results[1][0].page_content
    assert results[0][1] == 0.0
    assert results[1][1] == 0.5
    
    mock_graph_store.query.assert_called_once_with("test query")
    mock_vector_factory.search_best.assert_called_once_with("test query", k=4)

def test_vector_store_factory_list_indexes(mock_llm_provider):
    settings = MagicMock()
    settings.vector_store_base_path = MagicMock()
    settings.vector_store_base_path.exists.return_value = True
    
    with patch("os.listdir") as mock_listdir:
        with patch("pathlib.Path.is_dir") as mock_is_dir:
            mock_listdir.return_value = ["index1", "index2", "not_a_dir.txt"]
            mock_is_dir.side_effect = [True, True, False]
            
            factory = VectorStoreFactory(settings, mock_llm_provider)
            indexes = factory.list_indexes()
            
            assert indexes == ["index1", "index2"]

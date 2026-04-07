import pytest
from unittest.mock import MagicMock
from Src.Interfaces.ILLMProvider import ILLMProvider
from Src.Interfaces.IGraphStore import IGraphStore
from Src.Interfaces.IVectorStore import IVectorStore
from Src.Interfaces.IDocumentLoader import IDocumentLoader

@pytest.fixture
def mock_llm_provider():
    mock = MagicMock(spec=ILLMProvider)
    mock.invoke.return_value = "Mocked LLM Response"
    mock.ainvoke.return_value = "Mocked async LLM Response"
    return mock

@pytest.fixture
def mock_graph_store():
    mock = MagicMock(spec=IGraphStore)
    mock.query.return_value = []
    return mock

@pytest.fixture
def mock_vector_store():
    mock = MagicMock(spec=IVectorStore)
    mock.similarity_search.return_value = []
    return mock

@pytest.fixture
def mock_document_loader():
    mock = MagicMock(spec=IDocumentLoader)
    mock.load.return_value = []
    return mock

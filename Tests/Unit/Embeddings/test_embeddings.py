import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from Src.Embeddings.embeddings import embed_file, create_faiss_index, load_faiss_index, add_documents, save_faiss_index

@pytest.fixture
def mock_docs():
    from langchain_core.documents import Document
    return [Document(page_content="Test data", metadata={"source": "test.pdf"})]

@pytest.mark.asyncio
@patch("Src.Embeddings.embeddings.ingest_file", new_callable=AsyncMock)
@patch("Src.Embeddings.embeddings.create_faiss_index", new_callable=AsyncMock)
@patch("Src.Embeddings.embeddings.save_faiss_index", new_callable=AsyncMock)
@patch("os.path.exists")
@patch("Src.Utils.llm_utils.get_llm")
@patch("Src.Utils.llm_utils.setup_neo4j")
async def test_embed_file_new_index(mock_setup_neo4j, mock_get_llm, mock_exists, mock_save, mock_create, mock_ingest, mock_docs):
    # Setup mocks
    mock_ingest.return_value = mock_docs
    mock_exists.return_value = False
    mock_index = MagicMock()
    mock_create.return_value = mock_index
    mock_save.return_value = mock_index
    
    # Mock LLM and Graph extraction to avoid heavy lifting
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    with patch("langchain_experimental.graph_transformers.LLMGraphTransformer") as mock_transformer:
        mock_trans_instance = MagicMock()
        mock_trans_instance.convert_to_graph_documents.return_value = [] # No graph docs for simplicity
        mock_transformer.return_value = mock_trans_instance
        
        # Execute
        result = await embed_file("test.pdf")
        
        # Verify
        assert result == mock_index
        mock_ingest.assert_called_once_with("test.pdf")
        mock_create.assert_called_once()
        mock_save.assert_called_once()

@pytest.mark.asyncio
@patch("Src.Embeddings.embeddings.FAISS.from_documents")
async def test_create_faiss_index(mock_faiss_from, mock_docs):
    mock_index = MagicMock()
    mock_faiss_from.return_value = mock_index
    
    result = await create_faiss_index(mock_docs, "dummy_path", MagicMock())
    
    assert result == mock_index
    mock_faiss_from.assert_called_once()

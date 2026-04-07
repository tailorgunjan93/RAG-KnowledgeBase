"""
PyPDFDocumentLoader — Concrete implementation of IDocumentLoader for PDF files.

To add DOCX support: create DocxDocumentLoader(IDocumentLoader) and update container.py.
To add web scraping: create WebDocumentLoader(IDocumentLoader) and update container.py.
"""
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Src.Interfaces.IDocumentLoader import IDocumentLoader


class PyPDFDocumentLoader(IDocumentLoader):
    """
    Loads and chunks PDF documents using LangChain's PyPDFLoader.

    Chunk strategy:
        chunk_size=1000 characters with 200-character overlap — matches the
        embedding model's optimal context window for MiniLM-L6-v2.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def load(self, file_path: str) -> list[Document]:
        """
        Load a PDF file and return chunked Document objects.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            List of chunked LangChain Document objects.
        """
        loader = PyPDFLoader(file_path=str(file_path))
        documents = loader.load()
        return self._splitter.split_documents(documents)

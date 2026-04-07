"""
IDocumentLoader — Abstract contract for any document ingestion source.

Plug-and-Play: To support DOCX, HTML, or Unstructured.io, create a new class
that implements this interface. Only container.py needs to change.
"""
from abc import ABC, abstractmethod
from langchain_core.documents import Document


class IDocumentLoader(ABC):
    """
    Abstract base for all document loading/parsing providers.
    Implementations: PyPDFDocumentLoader, UnstructuredDocumentLoader, etc.
    """

    @abstractmethod
    async def load(self, file_path: str) -> list[Document]:
        """
        Load and chunk a document from the given file path.

        Args:
            file_path: Absolute path to the source document.

        Returns:
            List of chunked LangChain Document objects, ready for embedding.
        """
        ...

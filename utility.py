from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder


# Class: UtilityService
# ---------------------
# Centralized service providing access to core AI components:
# - Local Hugging Face chat generator (LLM)
# - SentenceTransformer-based embedders for documents and text
# - Chroma vector store and retriever setup
#
# This class ensures a consistent interface for embedding, retrieval,
# and generation across the entire Haystack-based RAG pipeline.
class UtilityService:
    def __init__(self) -> None:
        # Initialize Local LLM for conversational generation
        # Model: Qwen2.5-1.5B-Instruct
        self._llm = HuggingFaceLocalChatGenerator(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            generation_kwargs={"max_new_tokens": 150}
        )

        # Initialize Document Embedder for encoding large text documents
        self._docmument_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize Text Embedder for query or short text encoding
        self._text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            progress_bar=False
        )

    # Function: chroma_store
    # ----------------------
    # Creates or connects to a local persistent Chroma document store.
    #
    # Returns:
    #   ChromaDocumentStore: Persistent vector database instance for storage and retrieval.
    def chroma_store(self):
        return ChromaDocumentStore(
            collection_name="local_collection",
            persist_path='vectore_db/chroma_db'
        )

    # Function: chroma_retriever
    # ---------------------------
    # Provides a retriever for querying documents from the Chroma vector store.
    #
    # Parameters:
    #   top_k (int): Number of top-matching documents to return (default = 3)
    #
    # Returns:
    #   ChromaEmbeddingRetriever: Retriever instance for semantic search.
    def chroma_retriever(self, top_k: int = 3):
        return ChromaEmbeddingRetriever(
            document_store=self.chroma_store(),
            top_k=top_k
        )

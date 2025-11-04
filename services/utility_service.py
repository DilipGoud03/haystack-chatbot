from haystack_integrations.document_stores.weaviate.document_store import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.components.embedders.google_genai import GoogleGenAIDocumentEmbedder
from haystack_integrations.components.embedders.google_genai import GoogleGenAITextEmbedder
from dotenv import load_dotenv
import os
from decouple import config


# Class: UtilityService
# ---------------------
# Centralized service providing access to core AI components:
# - Google Gemini chat generator (LLM)
# - gemini-based embedders for documents and text
# - Weaviate vector store and retriever setup
#
# This class ensures a consistent interface for embedding, retrieval,
# and generation across the entire Haystack-based RAG pipeline.


class UtilityService:
    def __init__(self):
        load_dotenv()

        os.makedirs(str(config('DOC_DIR')), exist_ok=True)
        os.makedirs(str(config("PNG_DIR")), exist_ok=True)

        # Initialize Local LLM for conversational generation
        # Model: gemini-2.5-flash
        self._llm = GoogleGenAIChatGenerator(
            model="gemini-2.5-flash")

        # Initialize Document Embedder for encoding large text documents
        self._docmument_embedder = GoogleGenAIDocumentEmbedder(
            model="text-embedding-004")

        # Initialize Text Embedder for query or short text encoding
        self._text_embedder = GoogleGenAITextEmbedder(
            model="text-embedding-004")

    # Function: weaviate_store
    # ----------------------
    # Creates or connects to a local persistent weaviate document store.
    #
    # Returns:
    #   WeaviateDocumentStore: Persistent vector database instance for storage and retrieval.
    def weaviate_store(self):
        return WeaviateDocumentStore(url="http://localhost:8080")

    # Function: weaviate_retriever
    # ---------------------------
    # Provides a retriever for querying documents from the weaviate vector store.
    #
    # Parameters:
    #   top_k (int): Number of top-matching documents to return (default = 3)
    #
    # Returns:
    #   WeaviateEmbeddingRetriever: Retriever instance for semantic search.
    def weaviate_retriever(self, top_k: int = 3):
        return WeaviateEmbeddingRetriever(
            document_store=self.weaviate_store(),
            top_k=top_k
        )

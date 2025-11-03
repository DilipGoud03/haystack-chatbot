from haystack.tools import ComponentTool
from haystack import component
from typing import List, Dict, Any
from haystack.dataclasses import Document
from utility import UtilityService


# Initialize Utility Service
# --------------------------
# Provides access to core embedding models and retrievers.
utility_service = UtilityService()


# Class: RagSearcher
# ------------------
# Custom Haystack component that performs semantic retrieval (RAG Search)
# using Chroma vector storage and an embedding model.
#
# Workflow:
#   1. Encodes the input query using the text embedding model.
#   2. Searches the Chroma vector store for semantically similar documents.
#   3. Returns the top matching documents for further reasoning or summarization.
@component()
class RagSearcher:
    def __init__(self):
        # Initialize the text embedder and retriever
        self.text_embedder = utility_service._text_embedder
        self.retriever = utility_service.chroma_retriever(top_k=3)

    # Method: run
    # -----------
    # Executes the semantic search over the Chroma document store.
    #
    # Parameters:
    #   text (str): The input query or sentence to search for.
    #
    # Returns:
    #   Dict[str, Any]: A dictionary containing a list of matching `Document` objects.
    @component.output_types(documents=List[Document])
    def run(self, text: str) -> Dict[str, Any]:
        # Step 1: Generate text embeddings for the query
        emb_out = self.text_embedder.run(text=text)

        # Step 2: Retrieve top-k relevant documents from vector database
        docs_out = self.retriever.run(query_embedding=emb_out["embedding"])

        # Step 3: Return the matched documents
        return {"documents": docs_out["documents"]}


# Tool: rag_tool
# --------------
# Registers `RagSearcher` as a Haystack ComponentTool,
# enabling integration with Agents and other pipeline components.
rag_tool = ComponentTool(
    component=RagSearcher(),
    name="rag_search",
    description="Semantic search over the document."
)

from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder


class ModelService:
    def __init__(self) -> None:
        self._llm = HuggingFaceLocalChatGenerator(model="Qwen/Qwen2.5-1.5B-Instruct", generation_kwargs={"max_new_tokens": 150})
        self._docmument_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        self._text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2", progress_bar=False)
    
    def chroma_store(self):
        return ChromaDocumentStore(
            collection_name="local_collection",
            persist_path='vectore_db/chroma_db'
        )
    
    def chroma_retriever(self, top_k: int = 3):
        return ChromaEmbeddingRetriever(document_store=self.chroma_store(), top_k=top_k)
    
    def get_all(self):
        store = self.chroma_store()
        all_docs = store.filter_documents()
        return all_docs
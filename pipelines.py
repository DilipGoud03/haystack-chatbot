from haystack import Pipeline, component
from pathlib import Path
from haystack.dataclasses import ChatMessage, Document
from haystack.components.builders import ChatPromptBuilder
from typing import List, Dict, Any
from utility import UtilityService

utility_service = UtilityService()


@component()
class RagSearcher:
    def __init__(self, top_k: int = 3):
        self.text_embedder = utility_service._text_embedder
        self.text_embedder.warm_up()
        self.retriever = utility_service.chroma_retriever(top_k=top_k)

    @component.output_types(documents=List[Document])
    def run(self, text: str) -> Dict[str, Any]:
        emb_out = self.text_embedder.run(text=text)
        docs_out = self.retriever.run(query_embedding=emb_out["embedding"])
        return {"documents": docs_out["documents"]}


def search_from_document(query: str):
    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_user(
                """
                Answer the query based on the provided context.
                Also Answer will be a human readable.

                Context:
                {% for doc in documents %}
                {{ doc.content }}
                {% endfor %}

                Query: {{query}}
                Answer:
                """
            )
        ],
        required_variables="*"
    )

    # LLM (local HuggingFace model)
    llm = utility_service._llm
    llm.warm_up()

    retriever = utility_service.chroma_retriever()
    embedder = utility_service._text_embedder
    component = RagSearcher()
    # Build pipeline
    querying = Pipeline()
    # querying.add_component("embedder", embedder)
    # querying.add_component("retriever", retriever)
    querying.add_component('component', component)
    querying.add_component("prompt_builder", prompt_builder)
    querying.add_component("llm", llm)

    # Connect components
    # querying.connect("embedder.embedding", "retriever.query_embedding")
    querying.connect("component.documents", "prompt_builder.documents")
    querying.connect("prompt_builder.prompt", "llm.messages")

    # querying.draw(path='png' / Path('query.png'))
    print(querying.inputs())
    result = querying.run(
        {
            "component": {
                "text": query
            },
            "prompt_builder": {
                "query": query
            }
        }
    )

    return result

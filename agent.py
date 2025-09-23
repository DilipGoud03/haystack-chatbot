from haystack import component
from haystack.tools import ComponentTool
from typing import List, Dict, Any
from haystack.components.websearch import SerperDevWebSearch
import os
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, Document
from models import ModelService

model_service = ModelService()

SERPER_DEV_API_KEY = ''

if 'SERPERDEV_API_KEY' not in os.environ:
    os.environ['SERPERDEV_API_KEY'] = SERPER_DEV_API_KEY


@component()
class RagSearcher:
    def __init__(self, top_k: int = 3):
        self.retriever = model_service.chroma_retriever(top_k=top_k)
        self.text_embedder = model_service._text_embedder

    @component.output_types(documents=List[Document])
    def run(self, text: str) -> Dict[str, Any]:
        emb_out = self.text_embedder.run(text=text)
        docs_out = self.retriever.run(query_embedding=emb_out["embedding"])
        return {"documents": docs_out["documents"]}


rag_tool = ComponentTool(
    component=RagSearcher(),
    name="rag_search",
    description="Searches the local knowledge base using semantic embeddings to find relevant documents."
)


web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=3),
    name="web_search",
    description="Search the web using Serper.dev and return relevant documents."
)


def search_by_agent(query: str):
    system_prompt = """
    You are a helpful assistant.
    - Use rag_search first to retrieve information from the knowledge base.
    - Use web_search only when the query requires fresh, real-time, or external information (e.g., weather, breaking news).
    """
    agent = Agent(
        chat_generator=model_service._llm,
        system_prompt=system_prompt,
        tools=[rag_tool, web_tool],
    )
    agent.warm_up()
    msg = ChatMessage.from_user(query)
    resp = agent.run(messages=[msg])
    return resp


if __name__ == '__main__':
    while True:
        print('--'*50)
        question = input('Enter your query (or exit to quit) :')
        print('--'*50, '\n')
        if question.lower() == 'exit':
            break
        result = search_by_agent(question)
        print('\n', '--'*50)
        print(result["messages"][-1].text)
        print('--'*50, '\n')

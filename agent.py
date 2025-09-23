from haystack import component
from haystack.tools import ComponentTool
from haystack.core.super_component import SuperComponent
from typing import List, Dict, Any
from haystack.components.websearch import SerperDevWebSearch
import os
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, Document
from pipelines import UtilityService, search_pipeline  # Fixed import

utility_service = UtilityService()

SERPER_DEV_API_KEY = ''

if 'SERPERDEV_API_KEY' not in os.environ and SERPER_DEV_API_KEY:
    os.environ['SERPERDEV_API_KEY'] = SERPER_DEV_API_KEY


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


rag_tool = ComponentTool(
    component=RagSearcher(),
    name="rag_search",
    description="Semantic search over the company employee information knowledge base."
)


web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=3),
    name="web_search",
    description="Search the web using Serper.dev and return relevant documents."
)


def search_by_agent(query: str):
    system_prompt = """
    You are a helpful assistant.
    - ALWAYS call the rag_search tool first for every query, before responding.
    - Use web_search only when the knowledge base contains no relevant information or the user explicitly asks for real-time/external data.
    - If rag_search returns results, you must base your answer on them.
    """
    generator = utility_service._llm
    generator.warm_up()
    agent = Agent(
        chat_generator=generator,
        system_prompt=system_prompt,
        tools=[rag_tool, web_tool],
    )
    agent.warm_up()
    msg = ChatMessage.from_user(query)
    resp = agent.run(messages=[msg])
    print("Tools invoked →", tools_used(resp))
    return resp


def tools_used(run_output: dict) -> list[str]:
    seen, ordered = set(), []

    for msg in run_output["messages"]:
        for call in msg.tool_calls:
            if call.tool_name not in seen:
                ordered.append(call.tool_name)
                seen.add(call.tool_name)
    return ordered


if __name__ == '__main__':
    while True:
        print('--'*50)
        question = input('Enter your Question (or exit to quit): ')
        print('--'*50, '\n')
        if question.lower() == 'exit':
            break
        result = search_by_agent(question)
        print('\n', '--'*50)
        print(result["messages"][-1].text)
        print('--'*50, '\n')

from haystack import Pipeline
from pathlib import Path
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from models import ModelService
model_service = ModelService()


def search_query(query: str):
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
    llm = model_service._llm
    llm.warm_up()

    retriever = model_service.chroma_retriever()
    embedder = model_service._text_embedder

    # Build pipeline
    querying = Pipeline()
    querying.add_component("embedder", embedder)
    querying.add_component("retriever", retriever)
    querying.add_component("prompt_builder", prompt_builder)
    querying.add_component("llm", llm)

    # Connect components
    querying.connect("embedder.embedding", "retriever.query_embedding")
    querying.connect("retriever.documents", "prompt_builder.documents")
    querying.connect("prompt_builder.prompt", "llm.messages")

    querying.draw(path='png' / Path('query.png'))
    result = querying.run(
        {
            "embedder": {
                "text": query
            },
            "prompt_builder": {
                "query": query
            }
        }
    )

    return result


if __name__ == '__main__':
    while True:
        print('--'*50)
        question = input('Enter your query (or exit to quit) :')
        print('--'*50, '\n')
        if question.lower() == 'exit':
            break
        result = search_query(question)
        print('\n', '--'*50)
        print('Answer :', result["llm"]["replies"][0].text)
        print('--'*50, '\n')

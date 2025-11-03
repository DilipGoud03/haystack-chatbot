from haystack import Pipeline
from pathlib import Path
from haystack.dataclasses import ChatMessage
from haystack.components.builders import ChatPromptBuilder
from utility import UtilityService


# Initialize: Utility service instance for LLM and retriever setup
utility_service = UtilityService()


# Function: search_from_document
# ------------------------------
# Executes a retrieval-augmented query pipeline using Haystack.
# The function retrieves relevant documents, constructs a context-aware prompt,
# and generates a natural-language answer using the LLM.
def search_from_document(query: str):
    # Step 1: Build contextual prompt for the LLM
    prompt_builder = ChatPromptBuilder(
        template=[
            ChatMessage.from_user(
                """
                Answer the query based on the provided context.
                Also, ensure the answer is human-readable.

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

    # Step 2: Initialize model and retriever components
    llm = utility_service._llm
    llm.warm_up()
    retriever = utility_service.chroma_retriever()
    embedder = utility_service._text_embedder

    # Step 3: Construct the pipeline graph
    querying = Pipeline()
    querying.add_component("embedder", embedder)
    querying.add_component("retriever", retriever)
    querying.add_component("prompt_builder", prompt_builder)
    querying.add_component("llm", llm)

    # Step 4: Connect data flow between components
    querying.connect("embedder.embedding", "retriever.query_embedding")
    querying.connect("retriever.documents", "prompt_builder.documents")
    querying.connect("prompt_builder.prompt", "llm.messages")

    # Step 5: Generate and save pipeline visualization
    Path("png").mkdir(exist_ok=True)
    querying.draw(path=Path("png/query.png"))

    # Step 6: Execute query through the pipeline
    result = querying.run(
        {
            "embedder": {"text": query},
            "prompt_builder": {"query": query}
        }
    )

    return result


# Entry Point: CLI interactive mode
# ---------------------------------
# Enables manual testing of the pipeline by entering questions in a terminal loop.
if __name__ == '__main__':
    while True:
        print('--' * 50)
        question = input('Enter your query (or exit to quit): ')
        print('--' * 50, '\n')

        if question.lower() == 'exit':
            break

        result = search_from_document(question)
        print('\n', '--' * 50)
        print('Answer:', result["llm"]["replies"][0].text)
        print('--' * 50, '\n')

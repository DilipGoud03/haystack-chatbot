from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from tools.rag_search_tool import rag_tool
from tools.web_search_tool import web_tool
from services.utility_service import UtilityService
from haystack.components.generators.utils import print_streaming_chunk


# Initialize: Utility service and LLM generator
utility_service = UtilityService()
generator = utility_service._llm


# System Prompt
# -------------
# Defines the behavior and instruction set for the AI agent.
# The agent:
# - Uses RAG (Retrieval-Augmented Generation) for internal document search.
# - Falls back to Web Search when RAG yields no relevant results.
# - Produces clear and concise answers without explicitly mentioning tool usage.
system_prompt = """
    You are a helpful assistant.
    - Retrieve information using the appropriate tools.
    - If RAG search provides relevant results, use them to answer.
    - If no relevant information is found in RAG, use Web Search for real-time or external data.
    - Do not assume which tool was used — the system will append that automatically.
    - Always provide a clear and concise answer.
"""


# Create Agent
# -------------
# Combines the chat generator and both RAG + Web tools into a unified reasoning agent.
agent = Agent(
    chat_generator=generator,
    system_prompt=system_prompt,
    tools=[rag_tool, web_tool],
    streaming_callback=print_streaming_chunk
)


# Function: get_tool_result
# -------------------------
# Attempts to retrieve information for a query using:
#   1. RAG tool (for internal document retrieval)
#   2. Web tool (for external or real-time data)
# Returns:
#   - Tuple (result_text, tool_used)
#   - Falls back to "No relevant information found." if both tools fail
def get_tool_result(query: str):
    try:
        # Try RAG search first
        rag_result = rag_tool.invoke(text=query)
        documents = rag_result.get("documents", [])
        if documents and len(documents) > 0:
            text = "\n".join([doc.content for doc in documents])
            return text, "RAG"
    except Exception as e:
        print(f"RAG tool error: {e}")

    try:
        # Try Web search as fallback
        web_result = web_tool.invoke(query=query)
        text = web_result.get("results") or web_result.get(
            "content") or str(web_result)
        return text, "WEB"
    except Exception as e:
        print(f"Web tool error: {e}")

    # If neither tool succeeds
    return "No relevant information found.", "None"


# Entry Point: Interactive CLI mode
# ---------------------------------
# Allows the user to ask questions interactively via the console.
# The system automatically decides whether to use RAG or Web Search.
if __name__ == "__main__":
    while True:
        print("--" * 50)
        query = input("Please enter your query or (exit to quit): ")
        print("--" * 50, "\n")

        if query.lower() == "exit":
            break

        tool_output, tool_used = get_tool_result(query)

        user_msg = ChatMessage.from_user(query)
        response = agent.run(messages=[user_msg])

        print("--" * 50)
        print(f"Tool Used: {tool_used}")
        print(f"Answer: {response['messages'][-1].text}")
        print("--" * 50)

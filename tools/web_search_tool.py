from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch
import os


# Environment Setup
# -----------------
# Ensures the SERPERDEV_API_KEY is available for the web search component.
# If the key is not found in environment variables, a fallback is assigned.
if not os.environ.get("SERPERDEV_API_KEY"):
    os.environ["SERPERDEV_API_KEY"] =  os.getpass("Enter your SERPERDEV_API_KEY: ")


# Function: doc_to_string
# -----------------------
# Converts a list of Haystack `Document` objects into a formatted string
# representation. Primarily used to convert search results into readable
# text for downstream processing or display.
#
# Parameters:
#   documents (List[Document]): A list of documents containing metadata and content.
#
# Returns:
#   str: Concatenated text representation of the document contents.
#
# Notes:
#   - Each document includes a 'link' in metadata and 'content' as body text.
#   - Long text is truncated at 150,000 characters for safety.
def doc_to_string(documents) -> str:
    result_str = ""
    for document in documents:
        result_str += f"File Content for {document.meta['link']}\n\n {document.content}"

    if len(result_str) > 150_000:
        result_str = result_str[:150_000] + "...(large file can't be fully displayed)"

    return result_str


# Tool: web_tool
# --------------
# A Haystack `ComponentTool` wrapping the SerperDev Web Search component.
# This tool performs real-time web searches using the Serper API and returns
# top-k web documents for the given query.
#
# Parameters:
#   top_k (int): Number of search results to return (default = 5)
#
# Output:
#   The search results are processed into a single string via `doc_to_string`.
web_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=5),
    name="web_search",
    description="Search the web",
    outputs_to_string={"source": "documents", "handler": doc_to_string},
)

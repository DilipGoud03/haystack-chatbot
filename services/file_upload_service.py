import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.converters import TextFileToDocument, PyPDFToDocument, CSVToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from utility_service import UtilityService
from decouple import config

# Class: FileUploadService
# ------------------
# Custom Haystack component that performs semantic retrieval (RAG Search)
# using Chroma vector storage and an embedding model.
#
# Workflow:
#   1. Encodes the input query using the text embedding model.
#   2. Searches the Chroma vector store for semantically similar documents.
#   3. Returns the top matching documents for further reasoning or summarization.
class FileUploadService:
    def __init__(self):
        self.__utility_service = UtilityService()
        self.__doc_directory = str(config("DOC_DIR"))
        self.__png_directory = str(config("PNG_DIR"))

    # Function: upload_data
    # ---------------------
    # Builds and executes a Haystack indexing pipeline to process multiple document formats.
    # Features:
    #   - Handles TXT, PDF, and CSV file types.
    #   - Cleans, splits, embeds, and stores documents into Chroma DB.
    #   - Overwrites duplicates when re-indexing existing content.
    #   - Generates a visual diagram of the pipeline for debugging and documentation.
    def upload_data(self):
        # Define all document paths from the 'documents' folder
        file_paths = [self.__doc_directory / Path(name) for name in os.listdir(self.__doc_directory)]

        # Initialize Haystack components
        document_embedder = self.__utility_service._docmument_embedder
        document_writer = DocumentWriter(
            document_store=self.__utility_service.weaviate_store(),
            policy=DuplicatePolicy.OVERWRITE
        )
        document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
        document_cleaner = DocumentCleaner()
        text_converter = TextFileToDocument()
        csv_converter = CSVToDocument()
        pdf_converter = PyPDFToDocument()
        joiner = DocumentJoiner()
        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/csv"])

        # Build the indexing pipeline
        indexing_pipeline = Pipeline()
        indexing_pipeline.add_component('file_type_router', file_type_router)
        indexing_pipeline.add_component('text_converter', text_converter)
        indexing_pipeline.add_component('csv_converter', csv_converter)
        indexing_pipeline.add_component('pdf_converter', pdf_converter)
        indexing_pipeline.add_component('joiner', joiner)
        indexing_pipeline.add_component('cleaner', document_cleaner)
        indexing_pipeline.add_component('splitter', document_splitter)
        indexing_pipeline.add_component('embedder', document_embedder)
        indexing_pipeline.add_component('writer', document_writer)

        # Connect pipeline components (defining flow)
        indexing_pipeline.connect('file_type_router.text/plain', 'text_converter.sources')
        indexing_pipeline.connect('file_type_router.application/pdf', 'pdf_converter.sources')
        indexing_pipeline.connect('file_type_router.text/csv', 'csv_converter.sources')
        indexing_pipeline.connect('text_converter', 'joiner')
        indexing_pipeline.connect('pdf_converter', 'joiner')
        indexing_pipeline.connect('csv_converter', 'joiner')
        indexing_pipeline.connect('joiner', 'cleaner')
        indexing_pipeline.connect('cleaner', 'splitter')
        indexing_pipeline.connect('splitter', 'embedder')
        indexing_pipeline.connect('embedder', 'writer')

        # Optional: Generate a visual representation of the pipeline
        try:
            indexing_pipeline.draw(path=f'{self.__png_directory}/file_uploader.png')  # type: ignore
        except Exception as e:
            print(f"Warning: Failed to generate pipeline diagram: {e}")

        # Execute the pipeline on all documents
        indexing_pipeline.run({'file_type_router': {'sources': file_paths}})


def main():
    file_upload = FileUploadService()
    file_upload.upload_data()

# Entry Point: Run upload process
# -------------------------------
# When executed directly, runs the upload pipeline to process and index all documents.
if __name__ == '__main__':
    main()
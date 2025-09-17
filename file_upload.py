import os
from pathlib import Path
from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.converters import TextFileToDocument, PyPDFToDocument, CSVToDocument
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from models import ModelService
import os   


output_directory = 'png'
os.makedirs(output_directory, exist_ok=True)

_model_service = ModelService()


def upload_data():
    file_paths = ["data" / Path(name) for name in os.listdir("data")]
    document_embedder = _model_service._docmument_embedder
    document_writer = DocumentWriter(document_store=_model_service.chroma_store(), policy=DuplicatePolicy.OVERWRITE)
    document_splitter = DocumentSplitter(split_by="word", split_length=150, split_overlap=50)
    document_cleaner = DocumentCleaner()
    text_converter = TextFileToDocument()
    csv_converter = CSVToDocument()
    joiner = DocumentJoiner()
    pdf_converter = PyPDFToDocument()
    file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/csv"])
    

    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component('file_type_router', file_type_router)
    
    indexing_pipeline.add_component('text_converter', text_converter)
    indexing_pipeline.add_component('csv_converter', csv_converter)
    indexing_pipeline.add_component('pdf_converter', pdf_converter)
    
    indexing_pipeline.add_component('joiner', joiner)
    indexing_pipeline.add_component("cleaner", document_cleaner)
    indexing_pipeline.add_component("splitter", document_splitter)
    indexing_pipeline.add_component("embedder", document_embedder)
    indexing_pipeline.add_component("writer", document_writer)


    indexing_pipeline.connect('file_type_router.text/plain', 'text_converter.sources')
    indexing_pipeline.connect('file_type_router.application/pdf', 'pdf_converter.sources')
    indexing_pipeline.connect('file_type_router.text/csv', 'csv_converter.sources')

    indexing_pipeline.connect('text_converter', 'joiner')
    indexing_pipeline.connect('pdf_converter', 'joiner')
    indexing_pipeline.connect('csv_converter', 'joiner')
    
    indexing_pipeline.connect('joiner', 'cleaner')
    indexing_pipeline.connect("cleaner", "splitter")
    indexing_pipeline.connect("splitter", "embedder")
    indexing_pipeline.connect("embedder", "writer")

    indexing_pipeline.draw(path = f'{output_directory}/file_uploader.png') #type:ignore
    indexing_pipeline.run({'file_type_router': {'sources': file_paths}})

if __name__ == '__main__':
    upload_data()


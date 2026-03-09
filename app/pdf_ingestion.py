from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.vector_db import get_db
from app.vector_db import build_hybrid_retriever
import os

def process_pdf(path):

    loader = PyPDFLoader(path)
    doc = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(doc)

    filename = os.path.basename(path)

    for chunk in chunks:
        chunk.metadata["source"] = filename

    db = get_db()

    db.delete(where={"source": filename})

    db.add_documents(chunks)
    build_hybrid_retriever()
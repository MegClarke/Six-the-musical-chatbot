"""This module contains functions and classes related to loading and manipulating documents."""

import json

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_documents(filename: str, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load and split documents from a JSON file.

    Args:
        filename (str): The path to the JSON file containing documents.

    Returns:
        List[Document]: A list of Document objects split into smaller chunks.
    """
    with open(filename, "r", encoding="utf-8") as file:
        loaded_docs = json.load(file)

    json_docs = []

    for doc in loaded_docs:
        text = doc["content"]
        metadata = {"title": doc["title"]}
        json_docs.extend([Document(metadata=metadata, page_content=text)])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    json_documents = text_splitter.split_documents(json_docs)

    return json_documents


def format_docs(docs):
    """Format a list of Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

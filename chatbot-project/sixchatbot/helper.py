"""This module contains helper functions used in core.py."""

import json
import os

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def persist_directory_exists(persist_directory: str) -> bool:
    """Check if the persist directory exists and is not empty.

    Args:
        persist_directory (str): The directory where the vector store is persisted.

    Returns:
        bool: True if the directory exists and is not empty, False otherwise.
    """
    return os.path.exists(persist_directory) and os.listdir(persist_directory)


def get_documents(files: list[str], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load and split documents from a JSON file.

    Args:
        filenames (list[str]): A list of JSON filepaths containing documents.
        chunk_size (int): The size of the chunks to split the documents into.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        List[Document]: A list of Document objects split into smaller chunks.
    """
    json_docs = []

    for filename in files:
        with open(filename, "r", encoding="utf-8") as file:
            loaded_docs = json.load(file)

        for doc in loaded_docs:
            text = doc["content"]
            metadata = {"title": doc["title"]}
            json_docs.extend([Document(metadata=metadata, page_content=text)])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    json_documents = text_splitter.split_documents(json_docs)

    return json_documents


def format_docs(docs: list[Document]) -> str:
    """Format a list of Document objects into a single string.

    Args:
        docs (List[Document]): A list of Document objects.

    Returns:
        str: A string containing the contents of all the documents seperated by a blank line.
    """
    return "\n---\n".join(doc.page_content for doc in docs)

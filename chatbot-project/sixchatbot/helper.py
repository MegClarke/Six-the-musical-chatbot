"""This module contains helper functions used in core.py."""

import json
import os

from FlagEmbedding import FlagReranker
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


def get_json_files(directory: str) -> list[str]:
    """Get a list of JSON files from a directory.

    Args:
        directory (str): The directory path containing the JSON files.

    Returns:
        list[str]: A list of JSON filepaths.
    """
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, file)) and file.endswith(".json")
    ]


def get_documents(files: list[str], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Load and split documents from a JSON file.

    Args:
        filenames (list[str]): A list of JSON filepaths containing documents.
        chunk_size (int): The size of the chunks to split the documents into.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        list[Document]: A list of Document objects split into smaller chunks.
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


def rerank_context(context: list[Document], question: str, reranker: FlagReranker) -> list[Document]:
    """Rerank the context documents based on the question.

    Args:
        context (list[Document]): A list of Document objects.
        question (str): The question to rerank the context against.
        reranker (FlagReranker): The FlagReranker object to compute the scores.

    Returns:
        list[Document]: A list of Document objects sorted by the reranker scores.
    """
    paired_contexts = [[question, str(chunk)] for chunk in context]

    scores = reranker.compute_score(paired_contexts)
    scored_contexts = list(zip(scores, context, strict=True))
    scored_contexts.sort(reverse=True, key=lambda x: x[0])
    top_scored_contexts = scored_contexts[:10]

    return [chunk for _, chunk in top_scored_contexts]


def format_docs(docs: list[Document]) -> str:
    """Format a list of Document objects into a single string.

    Args:
        docs (list[Document]): A list of Document objects.

    Returns:
        str: A string containing the contents of all the documents seperated by a blank line.
    """
    return "\n---\n".join(doc.page_content for doc in docs)

"""This module sets up and runs a RAG chatbot with a knowledge base about "Six the Musical".

Exported Functions:
- get_documents(filename: str) -> List[Document]: Loads and splits documents from a JSON file.
- SingletonChroma: A singleton class for initializing and accessing the Chroma vector store.
- format_docs(docs: List[Document]) -> str: Formats a list of Document objects into a single string.
"""

from .core import load_config, persist_directory_exists, get_documents, get_retriever, format_docs

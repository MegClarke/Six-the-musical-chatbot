"""This module sets up and runs a RAG chatbot with a knowledge base about "Six the Musical".

Exported Functions from core.py:
- load_config(config_file: str) -> dict: Loads configuration from a YAML file.
- initialize_vector_store(files: list[str], config: Config): Initializes the vector store.
- update_vector_store(files: list[str], config: Config): Updates the vector store.
- get_retriever(
    persist_directory: str,
    search_kwargs: dict
  ) -> Chroma: Gets the retriever of the vector store in persist_directory.
- process_question(
    question: str,
    retriever: Chroma,
    prompt: PromptTemplate,
    llm: ChatOpenAI
  ) -> tuple[str, str]: Processes a question to generate a response.
- process_question_async(
    question: str,
    retriever: Chroma,
    prompt: PromptTemplate,
    llm: ChatOpenAI
  ) -> AsyncGenerator[str, None]: Processes a question to generate a response asynchronously.

Exported Functions from helper.py (ONLY USED FOR TEST CASES):
- persist_directory_exists(persist_directory: str) -> bool: Checks if the persist directory exists.
- get_files(directory_path: str) -> list[str]: Gets the files in a directory.
- get_documents(files: list[str]) -> list[Document]: Gets the documents from a list of files.

Exported Class from gsheets.py:
- QASheet: A class for interacting with the Q&A Google Sheets database.

Exported Class from schema.py:
- Config: A dataclass representing the configuration schema.
"""

from .core import (
    load_config,
    initialize_vector_store,
    update_vector_store,
    get_retriever,
    process_question,
    process_question_async,
)
from .helper import persist_directory_exists, get_json_files, get_documents
from .gsheets import QADatabase
from .schema import Config

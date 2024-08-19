"""This module sets up and runs a RAG chatbot with a knowledge base about "Six the Musical".

Exported Functions from core.py:
- load_config(config_file: str) -> dict: Loads configuration from a YAML file.
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
from .helper import persist_directory_exists, get_files, get_documents
from .gsheets import QADatabase
from .schema import Config

"""This module sets up and runs a RAG chatbot with a knowledge base about "Six the Musical".

Exported Functions:
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
- get_questions() -> list[str] | None: Fetches questions from a Google Sheet.
- post_chunks(data: list[str]) -> dict[str] | None: Posts retrieved chunks to a Google Sheet.
- post_answers(data: list[str]) -> dict[str] | None: Posts answers to a Google Sheet.
"""

from .core import load_config, persist_directory_exists, get_documents, get_retriever, format_docs, process_question
from .gsheets import get_questions, post_chunks, post_answers

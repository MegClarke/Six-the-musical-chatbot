"""This module initializes a vector store for the chatbot with the relevant documents from 'documents.json'."""


from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import sixchatbot


def initialize_vector_store(files: list[str], persist_directory: str, config: sixchatbot.Config):
    """Initialize the vector store.

    Args:
        files (list[str]): The list of files to load and split into documents
        persist_directory (str): The directory where the vector store is persisted.
        config (sixchatbot.Config): The configuration settings for the chatbot.
    """
    documents = sixchatbot.get_documents(files, config.text_splitter.chunk_size, config.text_splitter.chunk_overlap)

    Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=persist_directory,
    )
    print(f"Successfully initialized ChromaDB in {persist_directory!r}.")


def init():
    """Vector store initialization for the chatbot."""
    load_dotenv()
    config = sixchatbot.load_config()
    persist_directory = config.chroma.persist_directory

    if sixchatbot.persist_directory_exists(persist_directory):
        print(
            f"Chatbot has been previously initialized. \
              Persist directory {persist_directory!r} already contains data."
        )
        return

    files = ["documents.json", "tables.json"]
    initialize_vector_store(files, persist_directory, config)


if __name__ == "__main__":
    init()

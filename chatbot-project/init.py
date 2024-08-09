"""This module initializes a vector store for the chatbot with the relevant documents from 'documents.json'."""


from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import sixchatbot


def initialize_vector_store(persist_directory, config):
    """Initialize the vector store."""
    documents = sixchatbot.get_documents(
        "documents.json", config.text_splitter.chunk_size, config.text_splitter.chunk_overlap
    )

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
    initialize_vector_store(persist_directory, config)


if __name__ == "__main__":
    init()

"""This module initializes the chatbot by initializing the vector store with the relevant documents from 'documents.json'."""

import os

import yaml
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import sixchatbot


def init():
    """Vector store initialization for the chatbot."""
    load_dotenv()

    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    persist_directory = config["chroma"]["persist_directory"]

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(
            f"Chatbot has been previously initialized. \
              Persist directory {persist_directory!r} already contains data."
        )
        return
    documents = sixchatbot.get_documents(
        "documents.json", config["text_splitter"]["chunk_size"], config["text_splitter"]["chunk_overlap"]
    )

    Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=persist_directory,
    )
    print(f"Successfully initialized ChromaDB in {persist_directory!r}.")


if __name__ == "__main__":
    init()

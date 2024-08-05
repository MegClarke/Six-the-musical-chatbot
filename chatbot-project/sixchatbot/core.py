import os
from bs4 import SoupStrainer, BeautifulSoup
import json
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_documents(filename: str) -> list[Document]:
    """Load and split documents from a JSON file.

    Args:
        filename (str): The path to the JSON file containing documents.

    Returns:
        List[Document]: A list of Document objects split into smaller chunks.
    """
    with open(filename, 'r') as f:
        loaded_docs = json.load(f)

    json_docs = []

    for doc in loaded_docs:
        text = doc["content"]
        metadata = {"title": doc["title"]}
        json_docs.extend([Document(metadata=metadata, page_content=text)])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=200, add_start_index=True)
    json_documents = text_splitter.split_documents(json_docs)

    return json_documents


class SingletonChroma:
    """A singleton wrapper for the Chroma vector store."""
    _instance = None
    
    @staticmethod
    def get_instance(documents=None):
        """Get the singleton instance of SingletonChroma.

        Args:
            documents (Optional[List[Document]]): The documents to initialize the Chroma instance. Required if the instance is not yet created.

        Returns:
            SingletonChroma: The singleton instance.
        """
        if SingletonChroma._instance is None:
            if documents is None:
                raise ValueError("Documents must be provided for the first initialization.")
            SingletonChroma(documents)
        return SingletonChroma._instance

    def __init__(self, documents):
        """Initialize the SingletonChroma instance.

        Args:
            documents (List[Document]): The documents to initialize the Chroma instance.
        """
        if SingletonChroma._instance is not None:
            raise Exception("This class is a singleton!")
        SingletonChroma._instance = Chroma.from_documents(
            documents=documents,
            embedding=OpenAIEmbeddings(model="text-embedding-ada-002")
        )


def format_docs(docs):
    """Format a list of Document objects into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)



"""This module contains functions and classes used in init.py and main.py."""

import json
import os

import yaml
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .schema import Config


def load_config(config_file="config.yaml") -> dict:
    """Load configuration from a YAML file.

    Args:
        config_file (str): The path to the configuration file. Defaults to "config.yaml".

    Returns:
        dict: The configuration settings.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        return Config(**yaml.safe_load(file))


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

        if isinstance(loaded_docs, dict) and any(isinstance(v, dict) for v in loaded_docs.values()):
            for key, value in loaded_docs.items():
                text = json.dumps(value, indent=2)
                metadata = {"title": key}
                json_docs.append(Document(metadata=metadata, page_content=text))
        else:
            for doc in loaded_docs:
                text = doc["content"]
                metadata = {"title": doc["title"]}
                json_docs.extend([Document(metadata=metadata, page_content=text)])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    json_documents = text_splitter.split_documents(json_docs)

    return json_documents


def get_retriever(persist_directory: str, search_kwargs: dict) -> Chroma:
    """Get the retriever of the vector store in persist_directory.

    Args:
        persist_directory (str): The directory where the vector store is located.
        search_kwargs (dict): Search keyword arguments configured in config.yaml.

    Returns:
        Chroma: The ChromaDB retriever instance.
    """
    if persist_directory_exists(persist_directory):
        vector_store = Chroma(
            embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
            persist_directory=persist_directory,
            create_collection_if_not_exists=False,
        )
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    else:
        print("Make sure to run init.py before main.py. The vector store hasn't been initialized.")
        return None


def format_docs(docs: list[Document]) -> str:
    """Format a list of Document objects into a single string.

    Args:
        docs (List[Document]): A list of Document objects.

    Returns:
        str: A string containing the contents of all the documents seperated by a blank line.
    """
    return "\n---\n".join(doc.page_content for doc in docs)


def process_question(question: str, retriever: Chroma, prompt: PromptTemplate, llm: ChatOpenAI) -> tuple[str, str]:
    """Process a question by invoking the retriever and the RAG chain.

    Args:
        question (str): The question to process.
        retriever (Retriever): The retriever instance to use.
        prompt (PromptTemplate): The prompt template instance to use.
        llm (ChatOpenAI): The language model instance to use.

    Returns:
        tuple[str, str]: A tuple containing the retrieved context (chunks) and the generated response of the query.
    """
    """
    compressor = FlashrankRerank(top_n=8)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    """
    context = retriever.invoke(question)

    context_string = ""
    context_string = "\n\n".join(f"{str(chunk.metadata)}\n{chunk.page_content[:300]}" for chunk in context)

    context = format_docs(context)
    input_data = {"context": context, "question": question}

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(input_data)
    return context_string, response

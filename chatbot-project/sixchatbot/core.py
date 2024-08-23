"""This module contains functions and classes used in init.py and main.py."""

from typing import AsyncGenerator

import yaml
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .helper import format_docs, get_documents, get_json_files, persist_directory_exists, rerank_context
from .schema import Config


def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from a YAML file.

    Args:
        config_file (str): The path to the configuration file. Defaults to "config.yaml".

    Returns:
        dict: The configuration settings.
    """
    with open(config_file, "r", encoding="utf-8") as file:
        return Config(**yaml.safe_load(file))


def initialize_vector_store(files: list[str], config: Config) -> None:
    """Initialize the vector store.

    Args:
        files (list[str]): The list of files to load and split into documents
        config (Config): The configuration settings for the chatbot.
    """
    documents = get_documents(files, config.text_splitter.chunk_size, config.text_splitter.chunk_overlap)
    persist_directory = config.chroma.persist_directory
    embedding_function = OpenAIEmbeddings(model=config.chroma.embedding_model)

    Chroma.from_documents(
        documents=documents,
        embedding=embedding_function,
        persist_directory=persist_directory,
    )
    print(f"Successfully initialized ChromaDB in {persist_directory!r}.")


def update_vector_store(files: list[str], config: Config) -> None:
    """Update the vector store with new documents.

    Args:
        files (list[str]): The list of files to load and split into documents.
        config (sixchatbot.Config): The configuration settings for the chatbot.
    """
    persist_directory = config.chroma.persist_directory

    if persist_directory_exists(persist_directory) is False:
        initialize_vector_store(files, config)
        return

    new_documents = get_documents(files, config.text_splitter.chunk_size, config.text_splitter.chunk_overlap)

    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model=config.chroma.embedding_model),
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
    )

    vector_store.add_documents(new_documents)

    print(f"Successfully updated ChromaDB in {persist_directory!r}.")


def get_retriever(config: Config) -> Chroma:
    """Get the retriever of the vector store in persist_directory.

    Args:
        config (Config): The configuration settings for the chatbot.

    Returns:
        Chroma: The ChromaDB retriever instance.
    """
    persist_directory = config.chroma.persist_directory
    embedding_function = OpenAIEmbeddings(model=config.chroma.embedding_model)
    search_kwargs = config.search_kwargs
    directory_path = config.context_directory

    file_paths = get_json_files(directory_path)

    if persist_directory_exists(persist_directory) is False:
        initialize_vector_store(files=file_paths, config=config)
    vector_store = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
    )
    return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def process_question(
    question: str, retriever: Chroma, prompt: PromptTemplate, llm: ChatOpenAI, reranker: FlagReranker
) -> tuple[str, str]:
    """Process a question by invoking the retriever and the RAG chain.

    Args:
        question (str): The question to process.
        retriever (Retriever): The retriever instance to use.
        prompt (PromptTemplate): The prompt template instance to use.
        llm (ChatOpenAI): The language model instance to use.
        reranker (FlagReranker): The reranker instance to use.

    Returns:
        tuple[str, str]: A tuple containing the retrieved context (chunks) and the generated response of the query.
    """
    context = retriever.invoke(question)
    context = rerank_context(context, question, reranker)

    context_string = "\n\n".join(f"{str(chunk.metadata)}\n{chunk.page_content[:300]}" for chunk in context)

    context = format_docs(context)
    input_data = {"context": context, "question": question}

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke(input_data)
    return context_string, response


async def process_question_async(
    question: str, retriever: Chroma, prompt: PromptTemplate, llm: ChatOpenAI, reranker: FlagReranker
) -> AsyncGenerator[str, None]:
    """Process a question by invoking the retriever and the RAG chain with streaming.

    Args:
        question (str): The question to process.
        retriever (Retriever): The retriever instance to use.
        prompt (PromptTemplate): The prompt template instance to use.
        llm (ChatOpenAI): The language model instance to use.

    Yields:
        str: Chunks of the generated response.
    """
    context = retriever.invoke(question)
    context = rerank_context(context, question, reranker)

    context = format_docs(context)
    input_data = {"context": context, "question": question}

    rag_input = prompt.invoke(input_data)

    message = HumanMessage(content=str(rag_input))
    async for chunk in llm.astream([message]):
        yield chunk.content

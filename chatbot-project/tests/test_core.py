"""Test module for sixchatbot/core.py functions."""

from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from langchain.docstore.document import Document

from sixchatbot.core import (
    get_retriever,
    initialize_vector_store,
    load_config,
    process_question,
    process_question_async,
    update_vector_store,
)
from sixchatbot.schema import Config


@pytest.fixture
def mock_config(mock_load_config):
    """
    Fixture that provides a mock configuration object.

    Args:
        mock_load_config (fixture): Fixture for mocking the load_config function.

    Returns:
        Config: Mocked configuration object.
    """
    return mock_load_config.return_value


@pytest.fixture
def mock_documents():
    """
    Fixture that provides a list of mock documents.

    Returns:
        list: List of dictionaries, each representing a mock document.
    """
    return [{"text": "Sample document 1"}, {"text": "Sample document 2"}]


@patch("builtins.open", new_callable=mock_open, read_data="test_yaml_data")
@patch(
    "yaml.safe_load",
    return_value={
        "context_directory": "contexts",
        "reranker": {"model": "test-reranking-model"},
        "llm": {"model": "gpt-4o-mini", "temp": 0.001, "prompt": "prompts/default.txt"},
        "search_kwargs": {"k": 6},
        "chroma": {"persist_directory": "./chroma_persist", "embedding_model": "test-embedding-model"},
        "text_splitter": {"chunk_size": 1300, "chunk_overlap": 200},
    },
)
def test_load_config(mock_safe_load, mock_open):
    """
    Test the load_config function to ensure it loads the configuration from a YAML file correctly.

    Args:
        mock_safe_load (MagicMock): Mock for yaml.safe_load function.
        mock_open (MagicMock): Mock for builtins.open function.

    Asserts:
        The load_config function returns an instance of the Config class.
    """
    config = load_config("config.yaml")
    mock_safe_load.assert_called_once()
    assert isinstance(config, Config)


@patch("sixchatbot.core.get_documents")
@patch("sixchatbot.core.OpenAIEmbeddings")
@patch("sixchatbot.core.Chroma")
def test_initialize_vector_store(mock_chroma, mock_embeddings, mock_get_documents, mock_documents, mock_config):
    """
    Test the initialize_vector_store function to ensure it correctly initializes the vector store.

    Args:
        mock_get_documents (MagicMock): Mock for get_documents function.
        mock_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_documents (list): List of mock documents.
        mock_config (Config): Mock configuration object.

    Asserts:
        The initialize_vector_store function calls get_documents, OpenAIEmbeddings,
        and Chroma.from_documents with the correct arguments.
    """
    mock_files = ["file1.txt", "file2.txt"]
    mock_get_documents.return_value = mock_documents
    mock_embeddings.return_value = MagicMock()

    initialize_vector_store(mock_files, mock_config)

    mock_get_documents.assert_called_once_with(
        mock_files, mock_config.text_splitter.chunk_size, mock_config.text_splitter.chunk_overlap
    )
    mock_embeddings.assert_called_once_with(model=mock_config.chroma.embedding_model)
    mock_chroma.from_documents.assert_called_once_with(
        documents=mock_documents,
        embedding=mock_embeddings.return_value,
        persist_directory=mock_config.chroma.persist_directory,
    )


@patch("sixchatbot.core.persist_directory_exists", return_value=False)
@patch("sixchatbot.core.initialize_vector_store")
def test_update_vector_store_initializes_if_not_exists(
    mock_initialize_vector_store, mock_persist_directory_exists, mock_config
):
    """
    Test the update_vector_store function to ensure it initializes the vector store
    if the persist directory does not exist.

    Args:
        mock_initialize_vector_store (MagicMock): Mock for initialize_vector_store function.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_config (Config): Mock configuration object.

    Asserts:
        The initialize_vector_store function is called if the persist directory does not exist.
    """
    mock_files = ["file1.txt", "file2.txt"]

    update_vector_store(mock_files, mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    mock_initialize_vector_store.assert_called_once_with(mock_files, mock_config)


@patch("sixchatbot.core.get_documents")
@patch("sixchatbot.core.persist_directory_exists", return_value=True)
@patch("sixchatbot.core.Chroma")
@patch("sixchatbot.core.OpenAIEmbeddings")
def test_update_vector_store_adds_documents(
    mock_embeddings, mock_chroma, mock_persist_directory_exists, mock_get_documents, mock_documents, mock_config
):
    """
    Test the update_vector_store function to ensure it adds documents to an existing vector store.

    Args:
        mock_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_get_documents (MagicMock): Mock for get_documents function.
        mock_documents (list): List of mock documents.
        mock_config (Config): Mock configuration object.

    Asserts:
        Documents are added to an existing vector store if the persist directory exists.
    """
    mock_files = ["file1.txt", "file2.txt"]
    mock_get_documents.return_value = mock_documents
    mock_vector_store = MagicMock()
    mock_chroma.return_value = mock_vector_store

    update_vector_store(mock_files, mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    mock_get_documents.assert_called_once_with(
        mock_files, mock_config.text_splitter.chunk_size, mock_config.text_splitter.chunk_overlap
    )
    mock_chroma.assert_called_once_with(
        embedding_function=mock_embeddings.return_value,
        persist_directory=mock_config.chroma.persist_directory,
        create_collection_if_not_exists=False,
    )
    mock_vector_store.add_documents.assert_called_once_with(mock_documents)
    mock_embeddings.assert_called_once_with(model=mock_config.chroma.embedding_model)


@patch("sixchatbot.core.get_json_files", return_value=["file1.txt", "file2.txt"])
@patch("sixchatbot.core.persist_directory_exists", return_value=True)
@patch("sixchatbot.core.Chroma")
@patch("sixchatbot.core.OpenAIEmbeddings")
def test_get_retriever_existing_directory(
    mock_openai_embeddings, mock_chroma, mock_persist_directory_exists, mock_get_json_files, mock_config
):
    """
    Test the get_retriever function when the persist directory exists to ensure it correctly initializes the retriever.

    Args:
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_get_json_files (MagicMock): Mock for get_json_files function.
        mock_config (Config): Mock configuration object.

    Asserts:
        The get_retriever function initializes the retriever when the persist directory exists.
    """
    mock_chroma_instance = mock_chroma.return_value
    retriever_instance = MagicMock()
    mock_chroma_instance.as_retriever.return_value = retriever_instance

    retriever = get_retriever(mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    mock_chroma.assert_called_once_with(
        embedding_function=mock_openai_embeddings(model=mock_config.chroma.embedding_model),
        persist_directory=mock_config.chroma.persist_directory,
        create_collection_if_not_exists=False,
    )
    mock_chroma_instance.as_retriever.assert_called_once_with(
        search_type="similarity", search_kwargs={"k": mock_config.search_kwargs["k"]}
    )
    assert retriever == retriever_instance


@patch("sixchatbot.core.get_json_files", return_value=["file1.txt", "file2.txt"])
@patch("sixchatbot.core.persist_directory_exists", return_value=False)
@patch("sixchatbot.core.initialize_vector_store")
@patch("sixchatbot.core.Chroma")
@patch("sixchatbot.core.OpenAIEmbeddings")
def test_get_retriever_non_existing_directory(
    mock_openai_embeddings,
    mock_chroma,
    mock_initialize_vector_store,
    mock_persist_directory_exists,
    mock_get_json_files,
    mock_config,
):
    """
    Test the get_retriever function when the persist directory does not exist to ensure it initializes the vector store.

    Args:
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_initialize_vector_store (MagicMock): Mock for initialize_vector_store function.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_get_json_files (MagicMock): Mock for get_json_files function.
        mock_config (Config): Mock configuration object.

    Asserts:
        The get_retriever function initializes the vector store when the persist directory does not exist.
    """
    mock_files = mock_get_json_files.return_value
    get_retriever(mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    mock_chroma.assert_called_once_with(
        embedding_function=mock_openai_embeddings(model=mock_config.chroma.embedding_model),
        persist_directory=mock_config.chroma.persist_directory,
        create_collection_if_not_exists=False,
    )
    mock_initialize_vector_store.assert_called_once_with(files=mock_files, config=mock_config)


@patch("sixchatbot.core.FlagReranker")
@patch("sixchatbot.core.rerank_context")
@patch("sixchatbot.core.format_docs")
@patch("sixchatbot.core.StrOutputParser")
@patch("main.PromptTemplate")
@patch("main.ChatOpenAI")
def test_process_question(
    mock_llm, mock_prompt_template, mock_parser, mock_format_docs, mock_rerank_context, mock_flag_reranker
):
    """
    Test the process_question function to ensure it correctly processes a question using the retriever,
    reranker, and the RAG chain.

    Args:
        mock_format_docs (MagicMock): Mock for format_docs function.
        mock_flag_reranker (MagicMock): Mock for FlagReranker class.
        mock_llm (MagicMock): Mock for ChatOpenAI class.
        mock_prompt_template (MagicMock): Mock for PromptTemplate class.
        mock_parser (MagicMock): Mock for StrOutputParser class.
        mock_rerank_context (MagicMock): Mock for rerank_context function.

    Asserts:
        The function returns the expected context string and response.
    """
    # Mock inputs
    mock_question = "What is the capital of France?"
    mock_retriever = MagicMock()
    mock_prompt = mock_prompt_template.from_file.return_value
    mock_documents = [
        Document(page_content="Paris is the capital of France.", metadata={"source": "doc1"}),
        Document(page_content="Paris is the largest city in France.", metadata={"source": "doc2"}),
    ]

    mock_retriever.invoke.return_value = mock_documents

    # Mock the reranker behavior
    # mock_reranker_instance = mock_flag_reranker.return_value
    # mock_reranker_instance.compute_score.return_value = [0.9, 0.8]

    mock_rerank_context.return_value = mock_documents

    mock_format_docs.return_value = "Formatted documents."

    mock_rag_chain = mock_prompt | mock_llm | mock_parser
    mock_rag_chain_response = "Paris is the capital of France."
    mock_rag_chain.invoke.return_value = mock_rag_chain_response

    expected_input_data = {"context": mock_format_docs.return_value, "question": mock_question}

    # Mock the RAG chain behavior

    # Act
    context_string, response = process_question(
        mock_question, mock_retriever, mock_prompt, mock_llm, mock_flag_reranker
    )

    # Assert
    mock_retriever.invoke.assert_called_once_with(mock_question)
    mock_rerank_context.assert_called_once_with(mock_documents, mock_question, mock_flag_reranker)
    mock_format_docs.assert_called_once_with(mock_documents)
    mock_rag_chain.invoke.assert_called_once_with(expected_input_data)

    expected_context_string = (
        "{'source': 'doc1'}\nParis is the capital of France.\n\n"
        "{'source': 'doc2'}\nParis is the largest city in France."
    )
    assert context_string == expected_context_string
    assert response == mock_rag_chain_response


@pytest.mark.asyncio
@patch("sixchatbot.core.FlagReranker")
@patch("sixchatbot.core.rerank_context")
@patch("sixchatbot.core.format_docs")
@patch("main.ChatOpenAI")
async def test_process_question_async(mock_llm, mock_format_docs, mock_rerank_context, mock_flag_reranker):
    """
    Test the process_question_async function to ensure it processes a question correctly and streams the response.

    Args:
        mock_llm (AsyncMock): Mock for ChatOpenAI class.
        mock_format_docs (MagicMock): Mock for format_docs function.
        mock_flag_reranker (MagicMock): Mock for FlagReranker class.
    """
    # Mock inputs
    mock_question = "What is the capital of France?"
    mock_retriever = MagicMock()
    mock_prompt = MagicMock()
    mock_documents = [
        MagicMock(page_content="Paris is the capital of France.", metadata={"source": "doc1"}),
        MagicMock(page_content="Paris is the largest city in France.", metadata={"source": "doc2"}),
    ]
    mock_retriever.invoke.return_value = mock_documents

    mock_rerank_context.return_value = mock_documents

    mock_format_docs.return_value = "Formatted documents."

    # Mock the RAG chain behavior
    mock_llm_instance = mock_llm.return_value
    mock_llm_instance.astream.return_value = AsyncMock()
    mock_llm_instance.astream.return_value.__aiter__.return_value = iter(
        [MagicMock(content="Paris is the capital of France.")]
    )

    # Act
    response_generator = process_question_async(
        mock_question, mock_retriever, mock_prompt, mock_llm_instance, mock_flag_reranker
    )

    # Collect the streamed response
    response_chunks = []
    async for chunk in response_generator:
        response_chunks.append(chunk)

    # Assertions to verify correct calls
    mock_retriever.invoke.assert_called_once_with(mock_question)
    mock_rerank_context.assert_called_once_with(mock_documents, mock_question, mock_flag_reranker)
    mock_format_docs.assert_called_once_with(mock_documents)
    mock_llm_instance.astream.assert_called_once()

    assert response_chunks == ["Paris is the capital of France."]


@pytest.mark.asyncio
@patch("sixchatbot.core.FlagReranker")
@patch("sixchatbot.core.rerank_context")
@patch("sixchatbot.core.format_docs")
@patch("main.ChatOpenAI")
async def test_process_question_async_reranker_sorting(
    mock_llm, mock_format_docs, mock_rerank_context, mock_flag_reranker
):
    """
    Test the process_question_async function to ensure it correctly sorts the documents based on reranker scores.

    Args:
        mock_llm (AsyncMock): Mock for ChatOpenAI class.
        mock_format_docs (MagicMock): Mock for format_docs function.
        mock_flag_reranker (MagicMock): Mock for FlagReranker class.
    """
    # Mock inputs
    mock_question = "What is the capital of France?"
    mock_retriever = MagicMock()
    mock_prompt = MagicMock()
    mock_documents = [
        MagicMock(page_content="Paris is the capital of France.", metadata={"source": "doc1"}),
        MagicMock(page_content="Lyon is a city in France.", metadata={"source": "doc2"}),
        MagicMock(page_content="Marseille is a port city in France.", metadata={"source": "doc3"}),
    ]
    mock_retriever.invoke.return_value = mock_documents

    mock_rerank_context.return_value = mock_documents

    mock_format_docs.return_value = "Formatted documents."

    # Mock the RAG chain behavior
    mock_llm_instance = mock_llm.return_value
    mock_llm_instance.astream.return_value = AsyncMock()
    mock_llm_instance.astream.return_value.__aiter__.return_value = iter(
        [MagicMock(content="Lyon is a city in France.")]
    )

    # Act
    response_generator = process_question_async(
        mock_question, mock_retriever, mock_prompt, mock_llm_instance, mock_flag_reranker
    )

    # Collect the streamed response
    response_chunks = []
    async for chunk in response_generator:
        response_chunks.append(chunk)

    # Assertions to verify correct calls
    mock_retriever.invoke.assert_called_once_with(mock_question)
    mock_rerank_context.assert_called_once_with(mock_documents, mock_question, mock_flag_reranker)
    assert response_chunks == ["Lyon is a city in France."]

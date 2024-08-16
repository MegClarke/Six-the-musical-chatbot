"""Test module for sixchatbot/core.py functions."""

from unittest.mock import MagicMock, mock_open, patch

import pytest
from langchain.docstore.document import Document

from sixchatbot.core import get_retriever, initialize_vector_store, load_config, update_vector_store
from sixchatbot.helper import get_documents
from sixchatbot.schema import Config


@pytest.fixture
def mock_config(mock_load_config):
    return mock_load_config.return_value


@pytest.fixture
def mock_documents():
    """
    Provides a list of mock documents.

    Returns:
        list: List containing a mock document dictionary.
    """
    return [{"text": "Sample document 1"}, {"text": "Sample document 2"}]


@patch("builtins.open", new_callable=mock_open, read_data="test_yaml_data")
@patch(
    "yaml.safe_load",
    return_value={
        "context_directory": "contexts",
        "llm": {"model": "gpt-4o-mini", "temp": 0.001, "prompt": "prompts/default.txt"},
        "search_kwargs": {"k": 6},
        "chroma": {"persist_directory": "./chroma_persist"},
        "text_splitter": {"chunk_size": 1300, "chunk_overlap": 200},
    },
)
def test_load_config(mock_safe_load, mock_open):
    """
    Test the load_config function to ensure it loads configuration from a YAML file correctly.
    Args:
        mock_safe_load (MagicMock): Mock for yaml.safe_load function.
        mock_open (MagicMock): Mock for builtins.open function.
    """
    config = load_config("config.yaml")
    mock_safe_load.assert_called_once()
    assert isinstance(config, Config)


@patch("sixchatbot.get_documents")
@patch("sixchatbot.core.OpenAIEmbeddings")
@patch("sixchatbot.core.Chroma")
def test_initialize_vector_store(mock_chroma, mock_embeddings, mock_get_documents, mock_documents, mock_config):
    """
    Test the initialize_vector_store function to ensure it calls the appropriate methods
    with the correct arguments when initializing the vector store.

    Args:
        mock_get_documents (MagicMock): Mock for get_documents function.
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_from_documents (MagicMock): Mock for Chroma.from_documents method.
        mock_documents (list): List of mock documents.
        mock_config (Config): Mock configuration object.
    """
    mock_files = ["file1.txt", "file2.txt"]
    mock_get_documents.return_value = mock_documents
    mock_embeddings.return_value = MagicMock()

    # Act
    initialize_vector_store(mock_files, mock_config)

    # Assert
    mock_get_documents.assert_called_once_with(
        mock_files, mock_config.text_splitter.chunk_size, mock_config.text_splitter.chunk_overlap
    )
    mock_embeddings.assert_called_once_with(model="text-embedding-ada-002")
    mock_chroma.from_documents.assert_called_once_with(
        documents=mock_documents,
        embedding=mock_embeddings.return_value,
        persist_directory=mock_config.chroma.persist_directory,
    )


@patch("sixchatbot.initialize_vector_store")
@patch("sixchatbot.persist_directory_exists", return_value=False)
def test_update_vector_store_initializes_if_not_exists(
    mock_initialize_vector_store, mock_persist_directory_exists, mock_config
):
    mock_files = ["file1.txt", "file2.txt"]

    update_vector_store(mock_files, mock_config)

    mock_persist_directory_exists.assert_called_once_with("test_directory")
    mock_initialize_vector_store.assert_called_once_with(mock_files, mock_config)


@patch("sixchatbot.get_documents")
@patch("sixchatbot.persist_directory_exists", return_value=True)
@patch("sixchatbot.Chroma")
@patch("sixchatbot.OpenAIEmbeddings")
def test_update_vector_store_adds_documents(
    mock_embeddings, mock_chroma, mock_persist_directory_exists, mock_get_documents, mock_documents, mock_config
):
    # Arrange
    mock_files = ["file1.txt", "file2.txt"]
    mock_get_documents.return_value = mock_documents
    mock_vector_store = MagicMock()
    mock_chroma.return_value = mock_vector_store

    # Act
    update_vector_store(mock_files, mock_config)

    # Assert
    mock_persist_directory_exists.assert_called_once_with("test_directory")
    mock_get_documents.assert_called_once_with(mock_files, 100, 10)
    mock_chroma.assert_called_once_with(
        embedding_function=mock_embeddings.return_value,
        persist_directory="test_directory",
        create_collection_if_not_exists=False,
    )
    mock_vector_store.add_documents.assert_called_once_with(mock_documents)
    mock_embeddings.assert_called_once_with(model="text-embedding-ada-002")


@patch("sixchatbot.core.persist_directory_exists", return_value=True)
@patch("sixchatbot.core.Chroma")
@patch("sixchatbot.core.OpenAIEmbeddings")
def test_get_retriever_existing_directory(
    mock_openai_embeddings, mock_chroma, mock_persist_directory_exists, mock_config
):
    """
    Test the get_retriever function when the persist directory exists to ensure it corerectly initializes the retriever.
    Args:
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
    """
    mock_chroma_instance = mock_chroma.return_value
    retriever_instance = MagicMock()
    mock_chroma_instance.as_retriever.return_value = retriever_instance

    retriever = get_retriever(mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    mock_chroma.assert_called_once_with(
        embedding_function=mock_openai_embeddings(model="text-embedding-ada-002"),
        persist_directory=mock_config.chroma.persist_directory,
        create_collection_if_not_exists=False,
    )
    mock_chroma_instance.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 10})
    assert retriever == retriever_instance


@patch("sixchatbot.core.persist_directory_exists", return_value=False)
def test_get_retriever_non_existing_directory(mock_persist_directory_exists, mock_config):
    """
    Test the get_retriever function when the persist directory does not exist to ensure it returns None.
    Args:
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
    """

    retriever = get_retriever(mock_config)

    mock_persist_directory_exists.assert_called_once_with(mock_config.chroma.persist_directory)
    assert retriever is None

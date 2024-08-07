"""Test module for init.py which initializes the ChromaDB vector store."""

from unittest.mock import MagicMock, patch

import pytest

import init
import sixchatbot


@pytest.fixture
def mock_config():
    """
    Provides a mock configuration dictionary.

    Returns:
        dict: Mock configuration.
    """
    return {
        "text_splitter": {
            "chunk_size": 100,
            "chunk_overlap": 10,
        },
        "chroma": {"persist_directory": "test_directory"},
    }


@pytest.fixture
def mock_documents():
    """
    Provides a list of mock documents.

    Returns:
        list: List containing a mock document dictionary.
    """
    return [{"text": "Sample document"}]


@patch("sixchatbot.get_documents")
@patch("init.OpenAIEmbeddings")
@patch("init.Chroma.from_documents")
def test_initialize_vector_store(
    mock_from_documents, mock_openai_embeddings, mock_get_documents, mock_documents, mock_config
):
    """
    Test the initialize_vector_store function to ensure it calls the appropriate methods
    with the correct arguments when initializing the vector store.

    Args:
        mock_get_documents (MagicMock): Mock for get_documents function.
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_from_documents (MagicMock): Mock for Chroma.from_documents method.
        mock_config (dict): Mock configuration dictionary.
        mock_documents (list): List of mock documents.
    """
    mock_get_documents.return_value = mock_documents
    mock_openai_embeddings.return_value = MagicMock()

    init.initialize_vector_store("test_directory", mock_config)

    mock_get_documents.assert_called_once_with("documents.json", 100, 10)
    mock_openai_embeddings.assert_called_once_with(model="text-embedding-ada-002")
    mock_from_documents.assert_called_once_with(
        documents=mock_documents, embedding=mock_openai_embeddings.return_value, persist_directory="test_directory"
    )


@patch("init.load_dotenv")
@patch("init.initialize_vector_store")
@patch("sixchatbot.persist_directory_exists", return_value=True)
def test_init_already_initialized(
    mock_persist_directory_exists, mock_initialize_vector_store, mock_load_dotenv, mock_load_config
):
    """
    Test the init function when the persist directory already exists to ensure
    it does not call initialize_vector_store.

    Args:
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_initialize_vector_store (MagicMock): Mock for initialize_vector_store function.
        mock_load_dotenv (MagicMock): Mock for load_dotenv function.
        mock_load_config (MagicMock): Mock for load_config function. (declared in conftest.py)
    """

    init.init()

    mock_load_dotenv.assert_called_once()
    mock_load_config.assert_called_once()
    mock_persist_directory_exists.assert_called_once_with("test_directory")
    mock_initialize_vector_store.assert_not_called()


@patch("init.load_dotenv")
@patch("init.initialize_vector_store")
@patch("sixchatbot.persist_directory_exists", return_value=False)
def test_init_not_initialized(
    mock_persist_directory_exists, mock_initialize_vector_store, mock_load_dotenv, mock_load_config
):
    """
    Test the init function when the persist directory does not exist to ensure
    it calls initialize_vector_store with the correct arguments.

    Args:
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
        mock_initialize_vector_store (MagicMock): Mock for initialize_vector_store function.
        mock_load_dotenv (MagicMock): Mock for load_dotenv function.
        mock_load_config (MagicMock): Mock for load_config function. (declared in conftest.py)
    """

    init.init()

    mock_load_dotenv.assert_called_once()
    mock_load_config.assert_called_once()
    mock_persist_directory_exists.assert_called_once_with("test_directory")
    mock_initialize_vector_store.assert_called_once_with("test_directory", mock_load_config.return_value)

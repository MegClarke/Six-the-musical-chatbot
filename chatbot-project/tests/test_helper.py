"""Test module for sixchatbot/helper.py functions."""

import os
from unittest.mock import MagicMock, mock_open, patch

import pytest
from FlagEmbedding import FlagReranker
from langchain.docstore.document import Document

from sixchatbot.helper import format_docs, get_documents, get_json_files, persist_directory_exists, rerank_context


@patch("os.path.exists", return_value=True)
@patch("os.listdir", return_value=True)
def test_persist_directory_exists_true(mock_exists, mock_listdir):
    """
    Test the persist_directory_exists function to verify it returns True when the directory exists and is not empty.

    Args:
        mock_exists (MagicMock): Mock for os.path.exists function.
        mock_listdir (MagicMock): Mock for os.listdir function.

    Asserts:
        The function returns True when the directory exists and contains files.
    """
    result = persist_directory_exists("persist_directory")
    assert result is True


@patch("os.path.exists", return_value=False)
def test_persist_directory_exists_false(mock_exists):
    """
    Test the persist_directory_exists function to verify it returns False when the directory does not exist.

    Args:
        mock_exists (MagicMock): Mock for os.path.exists function.

    Asserts:
        The function returns False when the directory does not exist.
    """
    result = persist_directory_exists("persist_directory")
    assert result is False


@patch("os.listdir")
@patch("os.path.isfile")
def test_get_json_files_returns_json_files(mock_isfile, mock_listdir):
    """
    Test the get_json_files function to ensure it returns a list of JSON filepaths from the directory.

    Args:
        mock_isfile (MagicMock): Mock for os.path.isfile function.
        mock_listdir (MagicMock): Mock for os.listdir function.

    Asserts:
        The function returns only the files with a `.json` extension.
    """
    mock_listdir.return_value = ["file1.json", "file2.txt", "file3.json", "file4.md"]
    mock_isfile.side_effect = lambda filepath: not filepath.endswith(".md")

    expected_files = [os.path.join("test_directory", "file1.json"), os.path.join("test_directory", "file3.json")]

    result = get_json_files("test_directory")

    assert result == expected_files
    mock_listdir.assert_called_once_with("test_directory")


@patch("os.listdir", return_value=[])
def test_get_json_files_empty_directory(mock_listdir):
    """
    Test the get_json_files function to ensure it returns an empty list when the directory is empty.

    Args:
        mock_listdir (MagicMock): Mock for os.listdir function.

    Asserts:
        The function returns an empty list if there are no JSON files in the directory.
    """
    result = get_json_files("empty_directory")
    assert result == []
    mock_listdir.assert_called_once_with("empty_directory")


@patch("os.listdir", side_effect=FileNotFoundError)
def test_get_json_files_directory_not_found(mock_listdir):
    """
    Test the get_json_files function to handle the case when the directory does not exist.

    Args:
        mock_listdir (MagicMock): Mock for os.listdir function.

    Asserts:
        The function raises a FileNotFoundError if the directory does not exist.
    """
    with pytest.raises(FileNotFoundError):
        get_json_files("non_existent_directory")
    mock_listdir.assert_called_once_with("non_existent_directory")


@patch("os.listdir", return_value=["file1.json", "file2.json"])
@patch("os.path.isfile", return_value=False)
def test_get_json_files_no_valid_json_files(mock_isfile, mock_listdir):
    """
    Test the get_json_files function to ensure it returns an empty list when no valid JSON files are found.

    Args:
        mock_isfile (MagicMock): Mock for os.path.isfile function.
        mock_listdir (MagicMock): Mock for os.listdir function.

    Asserts:
        The function returns an empty list if there are no valid JSON files in the directory.
    """
    result = get_json_files("directory_with_invalid_files")
    assert result == []
    mock_listdir.assert_called_once_with("directory_with_invalid_files")
    mock_isfile.assert_any_call(os.path.join("directory_with_invalid_files", "file1.json"))
    mock_isfile.assert_any_call(os.path.join("directory_with_invalid_files", "file2.json"))


@patch("builtins.open", new_callable=mock_open, read_data='[{"content": "test content", "title": "test title"}]')
@patch("json.load", return_value=[{"content": "test content", "title": "test title"}])
@patch(
    "sixchatbot.core.RecursiveCharacterTextSplitter.split_documents",
    return_value=[Document(metadata={"title": "test title"}, page_content="test content")],
)
def test_get_documents(mock_split_documents, mock_json_load, mock_open):
    """
    Test the get_documents function to ensure it correctly loads, processes, and splits documents from a JSON file.

    Args:
        mock_split_documents (MagicMock): Mock for RecursiveCharacterTextSplitter.split_documents method.
        mock_json_load (MagicMock): Mock for json.load function.
        mock_open (MagicMock): Mock for builtins.open function.

    Asserts:
        The function correctly loads JSON content, splits it into documents, and returns the expected documents.
    """
    documents = get_documents(["file1.txt", "file2.txt"], chunk_size=1000, chunk_overlap=100)
    mock_json_load.assert_called()
    mock_split_documents.assert_called_once()
    assert len(documents) == 1
    assert documents[0].metadata["title"] == "test title"
    assert documents[0].page_content == "test content"


def test_rerank_context():
    """
    Test the rerank_context function to ensure it correctly reranks documents based on the question.

    Asserts:
        The function returns the documents in the correct order based on the reranker scores.
    """
    # Arrange: Mock inputs
    mock_question = "What is the capital of France?"
    mock_documents = [
        MagicMock(spec=Document, page_content="Paris is the capital of France.", metadata={"source": "doc1"}),
        MagicMock(spec=Document, page_content="Lyon is a city in France.", metadata={"source": "doc2"}),
        MagicMock(spec=Document, page_content="Marseille is a port city in France.", metadata={"source": "doc3"}),
    ]

    mock_reranker = MagicMock(spec=FlagReranker)
    mock_reranker.compute_score.return_value = [0.5, 0.9, 0.7]

    # Act: Call the function with the mocked inputs
    ranked_documents = rerank_context(mock_documents, mock_question, mock_reranker)

    # Assert: Check that the documents are reranked correctly
    assert len(ranked_documents) == 3
    assert ranked_documents[0].metadata["source"] == "doc2"
    assert ranked_documents[1].metadata["source"] == "doc3"
    assert ranked_documents[2].metadata["source"] == "doc1"

    # Check that the reranker was called with the correct arguments
    mock_reranker.compute_score.assert_called_once_with(
        [
            [mock_question, str(mock_documents[0])],
            [mock_question, str(mock_documents[1])],
            [mock_question, str(mock_documents[2])],
        ]
    )


def test_format_docs():
    """
    Test the format_docs function to ensure it correctly formats a list of documents into a single string.

    Asserts:
        The function returns the expected formatted string, combining document contents with separators.
    """
    docs = [
        Document(metadata={"title": "doc1"}, page_content="This is the content of doc1."),
        Document(metadata={"title": "doc2"}, page_content="This is the content of doc2."),
    ]
    formatted_docs = format_docs(docs)
    expected_output = "This is the content of doc1.\n---\nThis is the content of doc2."
    assert formatted_docs == expected_output

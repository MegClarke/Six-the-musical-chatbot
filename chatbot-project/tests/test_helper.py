"""Test module for sixchatbot/core.py functions."""

from unittest.mock import MagicMock, mock_open, patch

from langchain.docstore.document import Document

from sixchatbot.helper import format_docs, get_documents, persist_directory_exists


@patch("os.path.exists", return_value=True)
@patch("os.listdir", return_value=True)
def test_persist_directory_exists_true(mock_exists, mock_listdir):
    """
    Test the persist_directory_exists function when the directory exists to ensure it returns True.
    Args:
        mock_exists (MagicMock): Mock for os.path.exists function.
        mock_listdir (MagicMock): Mock for os.listdir function.
    """
    result = persist_directory_exists("persist_directory")
    assert result is True


@patch("os.path.exists", return_value=False)
def test_persist_directory_exists_false(mock_exists):
    """
    Test the persist_directory_exists function when the directory does not exist to ensure it returns False.
    Args:
        mock_exists (MagicMock): Mock for os.path.exists function.
    """
    result = persist_directory_exists("persist_directory")
    assert result is False


@patch("builtins.open", new_callable=mock_open, read_data='[{"content": "test content", "title": "test title"}]')
@patch("json.load", return_value=[{"content": "test content", "title": "test title"}])
@patch(
    "sixchatbot.core.RecursiveCharacterTextSplitter.split_documents",
    return_value=[Document(metadata={"title": "test title"}, page_content="test content")],
)
def test_get_documents(mock_split_documents, mock_json_load, mock_open):
    """
    Test the get_documents function to ensure it correctly processes and splits documents from a JSON file.
    Args:
        mock_split_documents (MagicMock): Mock for RecursiveCharacterTextSplitter.split_documents method.
        mock_json_load (MagicMock): Mock for json.load function.
        mock_open (MagicMock): Mock for builtins.open function.
    """
    documents = get_documents("filename.json", chunk_size=1000, chunk_overlap=100)
    mock_json_load.assert_called_once()
    mock_split_documents.assert_called_once()
    assert len(documents) == 1
    assert documents[0].metadata["title"] == "test title"
    assert documents[0].page_content == "test content"


def test_format_docs():
    """
    Test the format_docs function to ensure it correctly formats documents into a single string.
    """
    docs = [
        Document(metadata={"title": "doc1"}, page_content="This is the content of doc1."),
        Document(metadata={"title": "doc2"}, page_content="This is the content of doc2."),
    ]
    formatted_docs = format_docs(docs)
    expected_output = "This is the content of doc1.\n\nThis is the content of doc2."
    assert formatted_docs == expected_output

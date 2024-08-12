"""Test module for sixchatbot/core.py functions."""

from unittest.mock import MagicMock, mock_open, patch

from langchain.docstore.document import Document

from sixchatbot import Config, format_docs, get_documents, get_retriever, load_config, persist_directory_exists


@patch("builtins.open", new_callable=mock_open, read_data="test_yaml_data")
@patch(
    "yaml.safe_load",
    return_value={
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


@patch("sixchatbot.core.persist_directory_exists", return_value=True)
@patch("sixchatbot.core.Chroma")
@patch("sixchatbot.core.OpenAIEmbeddings")
def test_get_retriever_existing_directory(mock_openai_embeddings, mock_chroma, mock_persist_directory_exists):
    """
    Test the get_retriever function when the persist directory exists to ensure it corerectly initializes the retriever.
    Args:
        mock_openai_embeddings (MagicMock): Mock for OpenAIEmbeddings class.
        mock_chroma (MagicMock): Mock for Chroma class.
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
    """
    persist_directory = "mock_directory"
    search_kwargs = {"k": 10}
    mock_chroma_instance = mock_chroma.return_value
    retriever_instance = MagicMock()
    mock_chroma_instance.as_retriever.return_value = retriever_instance

    retriever = get_retriever(persist_directory, search_kwargs)

    mock_persist_directory_exists.assert_called_once_with(persist_directory)
    mock_chroma.assert_called_once_with(
        embedding_function=mock_openai_embeddings(model="text-embedding-ada-002"),
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
    )
    mock_chroma_instance.as_retriever.assert_called_once_with(search_type="similarity", search_kwargs={"k": 10})
    assert retriever == retriever_instance


@patch("sixchatbot.core.persist_directory_exists", return_value=False)
def test_get_retriever_non_existing_directory(mock_persist_directory_exists):
    """
    Test the get_retriever function when the persist directory does not exist to ensure it returns None.
    Args:
        mock_persist_directory_exists (MagicMock): Mock for persist_directory_exists function.
    """
    persist_directory = "mock_directory"
    search_kwargs = {"k": 10}
    retriever = get_retriever(persist_directory, search_kwargs)

    mock_persist_directory_exists.assert_called_once_with(persist_directory)
    assert retriever is None


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

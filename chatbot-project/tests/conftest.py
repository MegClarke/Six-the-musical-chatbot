"""This `conftest.py` file modifies the system path to help resolve the imports of modules in the test scripts."""

from unittest.mock import MagicMock, mock_open, patch

import pytest

import sixchatbot


@pytest.fixture
def mock_load_config():
    """Fixture that mocks the `load_config` function and returns a dictionary with mock values."""
    with patch("sixchatbot.load_config") as mock_load_config:
        mock_load_config.return_value = {
            "text_splitter": {
                "chunk_size": 100,
                "chunk_overlap": 10,
            },
            "chroma": {"persist_directory": "test_directory"},
            "search_kwargs": {"k": 10},
            "llm": {"model_name": "test_model", "prompt": "test_prompt.txt"},
        }
        yield mock_load_config

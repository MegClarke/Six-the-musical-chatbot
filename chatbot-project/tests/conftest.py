"""This `conftest.py` file modifies the system path to help resolve the imports of modules in the test scripts."""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sixchatbot import Config


@pytest.fixture
def mock_load_config():
    """Fixture that mocks the `load_config` function and returns a dictionary with mock values."""
    with patch("sixchatbot.load_config") as mock_load_config:
        mock_load_config.return_value = Config(
            context_directory="contexts",
            text_splitter={
                "chunk_size": 100,
                "chunk_overlap": 10,
            },
            chroma={"persist_directory": "test_directory"},
            search_kwargs={"k": 10},
            llm={
                "model": "test_model",
                "prompt": "test_prompt.txt",
                "temp": 0.001,
            },
        )
        yield mock_load_config

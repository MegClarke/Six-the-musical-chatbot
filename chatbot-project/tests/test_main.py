"""Test module for main.py which handles the main functionality of prompting the chatbot."""

from unittest.mock import MagicMock, patch

import pytest

import main
import sixchatbot


@patch("sixchatbot.get_retriever")
@patch("main.load_dotenv")
@patch("main.PromptTemplate")
@patch("main.ChatOpenAI")
def test_main(mock_chat_openai, mock_prompt_template, mock_load_dotenv, mock_get_retriever, mock_load_config):
    """
    Test the main function to ensure it initializes components and executes the RAG chain correctly.

    Args:
        mock_chat_openai (MagicMock): Mock for ChatOpenAI class.
        mock_prompt_template (MagicMock): Mock for PromptTemplate class.
        mock_load_dotenv (MagicMock): Mock for load_dotenv function.
        mock_get_retriever (MagicMock): Mock for get_retriever function.
        mock_load_config (MagicMock): Mock for load_config function. (declared in conftest.py)
    """

    mock_retriever_instance = MagicMock()
    mock_get_retriever.return_value = mock_retriever_instance
    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance
    mock_prompt_template_instance = MagicMock()
    mock_prompt_template.from_file.return_value = mock_prompt_template_instance

    # Run the main function
    main.main()

    # Assertions to verify correct calls
    mock_load_dotenv.assert_called_once()
    mock_load_config.assert_called_once()
    mock_get_retriever.assert_called_once_with("test_directory", {"k": 10})
    mock_chat_openai.assert_called_once_with(model_name="test_model")
    mock_prompt_template.from_file.assert_called_once_with("test_prompt.txt")

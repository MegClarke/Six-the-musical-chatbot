"""Test module for main.py which handles the main functionality of prompting the chatbot."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main


@patch("main.sixchatbot.QADatabase")
@patch("main.sixchatbot.process_question")
@patch("main.sixchatbot.get_retriever")
@patch("main.load_dotenv")
@patch("main.PromptTemplate")
@patch("main.ChatOpenAI")
@patch("main.sixchatbot.load_config")
@patch("os.getenv")
def test_main(
    mock_getenv,
    mock_load_config,
    mock_chat_openai,
    mock_prompt_template,
    mock_load_dotenv,
    mock_get_retriever,
    mock_process_question,
    mock_qadb,
):
    """
    Test the main function to ensure it initializes components, processes questions,
    and interacts with Google Sheets correctly.

    Args:
        mock_getenv (MagicMock): Mock for os.getenv function.
        mock_load_config (MagicMock): Mock for sixchatbot.load_config function.
        mock_chat_openai (MagicMock): Mock for ChatOpenAI class.
        mock_prompt_template (MagicMock): Mock for PromptTemplate class.
        mock_load_dotenv (MagicMock): Mock for load_dotenv function.
        mock_get_retriever (MagicMock): Mock for sixchatbot.get_retriever function.
        mock_process_question (MagicMock): Mock for sixchatbot.process_question function.
        mock_qadb (MagicMock): Mock for sixchatbot.QADatabase class.
    """
    # Mock instances
    mock_getenv.side_effect = ["test_spreadsheet_id", "test_trial"]
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config

    mock_retriever_instance = MagicMock()
    mock_get_retriever.return_value = mock_retriever_instance

    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance

    mock_prompt_template_instance = MagicMock()
    mock_prompt_template.from_file.return_value = mock_prompt_template_instance

    mock_qadb_instance = MagicMock()
    mock_qadb.return_value = mock_qadb_instance

    mock_qadb_instance.get_questions.return_value = ["Question 1", "Question 2"]
    mock_process_question.side_effect = [("Context 1", "Response 1"), ("Context 2", "Response 2")]

    # Run the main function
    main.main()

    # Assertions to verify correct calls
    mock_load_dotenv.assert_called_once()
    mock_load_config.assert_called_once()

    mock_get_retriever.assert_called_once_with(config=mock_config)
    mock_chat_openai.assert_called_once_with(model_name=mock_config.llm.model, temperature=mock_config.llm.temp)
    mock_prompt_template.from_file.assert_called_once_with(mock_config.llm.prompt)

    mock_qadb.assert_called_once_with(spreadsheet_id="test_spreadsheet_id", sheet_name="test_trial")
    mock_qadb_instance.get_questions.assert_called_once()

    # Ensure process_question was called for each question
    assert mock_process_question.call_count == 2
    mock_process_question.assert_any_call(
        "Question 1", mock_retriever_instance, mock_prompt_template_instance, mock_llm_instance
    )
    mock_process_question.assert_any_call(
        "Question 2", mock_retriever_instance, mock_prompt_template_instance, mock_llm_instance
    )

    # Ensure post_chunks and post_answers were called with the correct data
    mock_qadb_instance.post_chunks.assert_called_once_with(["Context 1", "Context 2"])
    mock_qadb_instance.post_answers.assert_called_once_with(["Response 1", "Response 2"])


@patch("main.sixchatbot.process_question_async")
@patch("main.sixchatbot.get_retriever")
@patch("main.load_dotenv")
@patch("main.PromptTemplate")
@patch("main.ChatOpenAI")
@patch("main.sixchatbot.load_config")
@pytest.mark.asyncio
async def test_query_chatbot(
    mock_load_config,
    mock_chat_openai,
    mock_prompt_template,
    mock_load_dotenv,
    mock_get_retriever,
    mock_process_question_async,
):
    """
    Test the query_chatbot function to ensure it initializes components and processes a question correctly.

    Args:
        mock_load_config (MagicMock): Mock for sixchatbot.load_config function.
        mock_chat_openai (MagicMock): Mock for ChatOpenAI class.
        mock_prompt_template (MagicMock): Mock for PromptTemplate class.
        mock_load_dotenv (MagicMock): Mock for load_dotenv function.
        mock_get_retriever (MagicMock): Mock for sixchatbot.get_retriever function.
        mock_process_question_async (AsyncMock): Mock for sixchatbot.process_question_async function.
    """
    # Mock instances
    mock_config = MagicMock()
    mock_load_config.return_value = mock_config

    mock_retriever_instance = MagicMock()
    mock_get_retriever.return_value = mock_retriever_instance

    mock_llm_instance = MagicMock()
    mock_chat_openai.return_value = mock_llm_instance

    mock_prompt_template_instance = MagicMock()
    mock_prompt_template.from_file.return_value = mock_prompt_template_instance

    # Mock the response from process_question_async
    mock_chunk = "Test response chunk"
    mock_process_question_async.return_value = AsyncMock()
    mock_process_question_async.return_value.__aiter__.return_value = iter([mock_chunk])

    # Run the query_chatbot function
    response_generator = main.query_chatbot("Test question")

    # Collect the streamed response
    response_chunks = []
    async for chunk in response_generator:
        response_chunks.append(chunk)

    # Assertions to verify correct calls
    mock_load_dotenv.assert_called_once()
    mock_load_config.assert_called_once()
    mock_get_retriever.assert_called_once_with(config=mock_config)
    mock_chat_openai.assert_called_once_with(
        model_name=mock_config.llm.model, temperature=mock_config.llm.temp, streaming=True
    )
    mock_prompt_template.from_file.assert_called_once_with(mock_config.llm.prompt)
    mock_process_question_async.assert_called_once_with(
        "Test question", mock_retriever_instance, mock_prompt_template_instance, mock_llm_instance
    )

    # Check the streamed response chunks
    assert response_chunks == [mock_chunk]

"""Test module for the FastAPI endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


@patch("main.query_chatbot")
def test_query_chatbot_endpoint(mock_query_chatbot):
    """Test the chatbot query endpoint."""
    # Arrange
    mock_query_chatbot.return_value = AsyncMock()
    mock_query_chatbot.return_value.__aiter__.return_value = iter(["response part 1\n", "response part 2\n"])

    question_input = {"question": "What is the meaning of life?"}

    # Act
    response = client.put("/query", json=question_input)

    # Assert
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    mock_query_chatbot.assert_called_once_with(question_input["question"])

    # Read and check the streaming response
    streamed_content = list(response.iter_lines())
    assert streamed_content == ["response part 1", "response part 2"]

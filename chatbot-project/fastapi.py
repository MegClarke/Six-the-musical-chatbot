"""FastAPI application for the chatbot project."""
import main
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    """Hello world endpoint for the FastAPI application."""
    return {"message": "Hello World"}


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
)
async def health():
    """Health check endpoint for the FastAPI application."""
    return {"status": "OK"}


@app.get(
    "/query/{question}",
    tags=["query"],
    summary="Query the Chatbot",
    response_description="Return the response of the chatbot",
)
async def query_chatbot(question: str):
    """Query the chatbot with a question."""
    return {"response": main.query_chatbot(question)}

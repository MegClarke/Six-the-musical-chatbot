"""FastAPI application for the chatbot project."""
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

import main

app = FastAPI()


@app.get("/")
async def root():
    """Redirects to the Swagger documentation."""
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
)
async def health():
    """Health check endpoint for the FastAPI application."""
    return {"status": "OK"}


class QuestionInput(BaseModel):
    """Pydantic model for the question input."""

    question: str


@app.post(
    "/query",
    tags=["query"],
    summary="Query the Chatbot",
    response_description="Stream the response of the chatbot",
)
async def query_chatbot_endpoint(question_input: QuestionInput):
    """Query the chatbot with a question."""
    return StreamingResponse(main.query_chatbot(question_input.question), media_type="text/plain")

"""This module handles the main functionality of prompting the chatbot."""

import os
from typing import AsyncGenerator

from dotenv import load_dotenv
from FlagEmbedding import FlagReranker
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import sixchatbot


async def query_chatbot(question: str) -> AsyncGenerator[str, None]:
    """Query the chatbot with a question.

    Args:
        question (str): The question to ask the chatbot.

    Yields:
        str: Chunks of the response from the chatbot.
    """
    load_dotenv()
    config = sixchatbot.load_config()
    retriever = sixchatbot.get_retriever(config=config)
    llm = ChatOpenAI(model_name=config.llm.model, temperature=config.llm.temp)
    reranker = FlagReranker(model_name_or_path=config.reranker.model, use_fp16=True)
    prompt = PromptTemplate.from_file(config.llm.prompt)

    async for chunk in sixchatbot.process_question_async(question, retriever, prompt, llm, reranker):
        yield chunk


def main() -> None:
    """Main function for the chatbot."""
    load_dotenv()
    config = sixchatbot.load_config()
    retriever = sixchatbot.get_retriever(config=config)
    llm = ChatOpenAI(model_name=config.llm.model, temperature=config.llm.temp)
    reranker = FlagReranker(model_name_or_path=config.reranker.model, use_fp16=True)
    prompt = PromptTemplate.from_file(config.llm.prompt)

    spreadsheet_id = os.getenv("SHEET_ID")
    sheet_name = os.getenv("SHEET_NAME")

    qa_db = sixchatbot.QADatabase(spreadsheet_id=spreadsheet_id, sheet_name=sheet_name)

    questions = qa_db.get_questions()
    retrieved_chunks = []
    outputs = []

    for question in questions:
        context_string, response = sixchatbot.process_question(question, retriever, prompt, llm, reranker)
        retrieved_chunks.append(context_string)
        outputs.append(response)

    qa_db.post_chunks(retrieved_chunks)
    qa_db.post_answers(outputs)


if __name__ == "__main__":
    main()

"""This module handles the main functionality of prompting the chatbot."""

import json
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import sixchatbot


def main() -> None:
    """Main function for the chatbot."""
    load_dotenv()

    config = sixchatbot.load_config()

    retriever = sixchatbot.get_retriever(config=config)

    llm = ChatOpenAI(model_name=config.llm.model, temperature=config.llm.temp)
    prompt = PromptTemplate.from_file(config.llm.prompt)

    spreadsheet_id = os.getenv("SHEET_ID")
    sheet_name = "Trial 6"

    qa_db = sixchatbot.QADatabase(spreadsheet_id=spreadsheet_id, sheet_name=sheet_name)

    questions = qa_db.get_questions()
    retrieved_chunks = []
    outputs = []

    for question in questions:
        context_string, response = sixchatbot.process_question(question, retriever, prompt, llm)
        retrieved_chunks.append(context_string)
        outputs.append(response)

    qa_db.post_chunks(retrieved_chunks)
    qa_db.post_answers(outputs)


if __name__ == "__main__":
    main()

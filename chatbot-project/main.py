"""This module handles the main functionality of prompting the chatbot."""

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import sixchatbot


def main():
    """Main function for the chatbot."""
    load_dotenv()
    config = sixchatbot.load_config()
    persist_directory = config.chroma.persist_directory
    search_kwargs = config.search_kwargs

    retriever = sixchatbot.get_retriever(persist_directory, search_kwargs)

    llm = ChatOpenAI(model_name=config.llm.model, temperature=config.llm.temp)
    prompt = PromptTemplate.from_file(config.llm.prompt)

    questions = sixchatbot.get_questions()
    retrieved_chunks = []
    outputs = []

    for question in questions:
        context_string, response = sixchatbot.process_question(question, retriever, prompt, llm)
        retrieved_chunks.append(context_string)
        outputs.append(response)

    print(retrieved_chunks)
    print(outputs)

    sixchatbot.post_chunks(retrieved_chunks)
    sixchatbot.post_answers(outputs)


if __name__ == "__main__":
    main()

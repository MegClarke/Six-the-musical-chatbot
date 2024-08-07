"""This module handles the main functionality of prompting the chatbot."""

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import sixchatbot


def main():
    """Main function for the chatbot."""
    load_dotenv()
    config = sixchatbot.load_config()
    persist_directory = config["chroma"]["persist_directory"]
    search_kwargs = config["search_kwargs"]

    retriever = sixchatbot.get_retriever(persist_directory, search_kwargs)

    llm = ChatOpenAI(model_name=config["llm"]["model_name"])
    prompt = PromptTemplate.from_file(config["llm"]["prompt"])

    rag_chain = (
        {"context": retriever | sixchatbot.format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke("Tell me about Anne Boleyn in Six the musical")
    print(response)


if __name__ == "__main__":
    main()

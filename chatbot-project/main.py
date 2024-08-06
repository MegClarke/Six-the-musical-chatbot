"""This module handles the main functionality of prompting the chatbot."""

import os

import yaml
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sixchatbot


def main():
    """Main function for the chatbot."""
    load_dotenv()

    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    persist_directory = config["chroma"]["persist_directory"]

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vector_store = Chroma(
            embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
            persist_directory=config["chroma"]["persist_directory"],
            create_collection_if_not_exists=False,
        )
    else:
        print("Make sure to run init.py before main.py. The vector store hasn't been initialized.")
        return

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": config["search_kwargs"]["k"]})

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

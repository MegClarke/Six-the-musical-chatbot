"""This module handles the main functionality of the chatbot."""

import yaml
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

import sixchatbot


def main():
    """Main function for the chatbot."""
    load_dotenv()

    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    documents = sixchatbot.get_documents(
        "documents.json", config["text_splitter"]["chunk_size"], config["text_splitter"]["chunk_overlap"]
    )
    singleton_chroma = sixchatbot.SingletonChroma.get_instance(documents=documents)
    retriever = singleton_chroma.as_retriever(
        search_type="similarity", search_kwargs={"k": config["search_kwargs"]["k"]}
    )

    llm = ChatOpenAI(model_name=config["llm"]["model_name"])
    prompt = PromptTemplate.from_file(config["llm"]["prompt"])

    rag_chain = (
        {"context": retriever | sixchatbot.format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = rag_chain.invoke("Who was Anne Boleyn in the context of Six the Musical?")
    print(response)


if __name__ == "__main__":
    main()

"""This module handles the main functionality of prompting the chatbot."""

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sixchatbot


def update_vector_store(files: list[str], persist_directory: str, config: sixchatbot.Config):
    """Update the vector store with new documents.

    Args:
        files (list[str]): The list of files to load and split into documents.
        persist_directory (str): The directory where the vector store is persisted.
        config (sixchatbot.Config): The configuration settings for the chatbot.
    """
    if sixchatbot.persist_directory_exists(persist_directory) is False:
        print(f"The persist directory is empty. Please initialize the vector store first with {files}.")
        return

    new_documents = sixchatbot.get_documents(files, config.text_splitter.chunk_size, config.text_splitter.chunk_overlap)

    vector_store = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=persist_directory,
        create_collection_if_not_exists=False,
    )

    vector_store.add_documents(new_documents)

    print(f"Successfully updated ChromaDB in {persist_directory!r}.")


def main():
    """Main function for the chatbot."""
    load_dotenv()
    config = sixchatbot.load_config()
    persist_directory = config.chroma.persist_directory
    search_kwargs = config.search_kwargs

    """
    files = ["tables.json"]
    update_vector_store(files, persist_directory, config)
    """

    retriever = sixchatbot.get_retriever(persist_directory, search_kwargs)

    llm = ChatOpenAI(model_name=config.llm.model, temperature=config.llm.temp)
    prompt = PromptTemplate.from_file(config.llm.prompt)

    sheet_name = "Trial 2"
    questions = sixchatbot.get_questions(sheet_name)
    retrieved_chunks = []
    outputs = []

    for question in questions:
        context_string, response = sixchatbot.process_question(question, retriever, prompt, llm)
        retrieved_chunks.append(context_string)
        outputs.append(response)

    sixchatbot.post_chunks(sheet_name, retrieved_chunks)
    sixchatbot.post_answers(sheet_name, outputs)


if __name__ == "__main__":
    main()

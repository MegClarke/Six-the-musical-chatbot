import os
from bs4 import SoupStrainer, BeautifulSoup
import json
from langchain import hub
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader,  WikipediaLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv


def get_documents(filename: str) -> list[Document]:
    with open(filename, 'r') as f:
        loaded_docs = json.load(f)

    print(loaded_docs)
    json_docs = []

    for doc in loaded_docs:
        text = doc["content"]
        metadata = {"title": doc["title"]}
        json_docs.extend([Document(metadata=metadata, page_content=text)])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1300, chunk_overlap=200, add_start_index=True)
    json_documents = text_splitter.split_documents(json_docs)

    return json_documents


if __name__ == "__main__":
    load_dotenv()

    documents = get_documents("documents.json")
    print("")

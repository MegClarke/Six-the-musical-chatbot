from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import sixchatbot


def main():
    load_dotenv()

    documents = sixchatbot.get_documents("documents.json")
    singleton_chroma = sixchatbot.SingletonChroma.get_instance(documents=documents)  
    retriever = singleton_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    #config 
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    prompt = hub.pull("rlm/rag-prompt")

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

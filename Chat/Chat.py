import platform
import textwrap

from colorama import Fore, Back, Style

if platform.system() != "Darwin":
    __import__('pysqlite3')
    import sys

    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv


def make_chain(model_name: str, vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Creates a Chroma vector store and persists the documents to disk.

    :return: A chain with the specified model using the vector DB for retrieval.
    """
    model = ChatOpenAI(
        model_name=model_name,
        temperature="0",
        verbose=True
    )

    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )


if __name__ == "__main__":
    load_dotenv()

    embedding = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="ethics-chunks",
        embedding_function=embedding,
        persist_directory="data/chroma",
    )

    chain = make_chain("gpt-3.5-turbo", vector_store)
    chat_history = []

    while True:
        print()
        question = input("Question: ")

        # Generate answer
        response = chain.invoke({"question": question, "chat_history": chat_history})

        # Parse answer
        answer = response["answer"]
        source = response["source_documents"]
        chat_history.append(HumanMessage(content=question))

        # Wrap answer in 160 character lines
        wrapped_text = textwrap.wrap(str(AIMessage(content=answer)), 160)
        chat_history.append(AIMessage(content=wrapped_text))

        # Display answer
        print("\n\nSources:\n")
        for document in source:
            print(f"Page: {document.metadata['page_number']}")
            print(f"Text chunk: {document.page_content[:160]}...\n\n")

        for line in chat_history:
            print(f"{Fore.YELLOW}{Back.BLACK}{Style.BRIGHT}{line}")

    # print(f"{Fore.YELLOW}{Back.BLACK}{Style.BRIGHT}Answer: {answer}")

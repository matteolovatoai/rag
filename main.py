from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from config import settings

def main():
    embeddings = OllamaEmbeddings(model=settings.embeddings_model)
    if settings.db_path.exists() and any(settings.db_path.iterdir()):
        print("Chroma database already exists. Skipping creation.")
        vectorstore = Chroma(persist_directory=str(settings.db_path), embedding_function=embeddings)
    else:
        print("Loading the PDF file...")
        reader = PdfReader(settings.pdf_path)
        docs = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"page": i})
                docs.append(doc)
        print("Splitting the text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        splits = text_splitter.split_documents(docs)
        print("Creating vectorial database...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=str(settings.db_path))
    
    print("Searching...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.search_kwargs})
    print("Configuring the LLM...")
    llm = ChatOllama(model="qwen2.5:3b")
    # rewriting the question to inject the history of the chat
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", settings.rag_contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", settings.rag_contextualize_q_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"), 
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    print("=" * 50)
    print("\nSISTEMA PRONTO!\n")
    print("Fai le tue domande sul PDF. Digita 'esci' per chiudere il programma.")
    print("=" * 50)

    chat_history = []

    while True:
        domanda = input("Domanda: ")
        if domanda == "esci":
            break

        print("Generazione della risposta in corso (potrebbe richiedere qualche secondo)...\n")

        # Interroghiamo il RAG
        response = rag_chain.invoke({"input": domanda, "chat_history": chat_history})

        chat_history.append(HumanMessage(content=domanda))
        chat_history.append(AIMessage(content=response["answer"]))

        if len(chat_history) > settings.rag_max_len_chat_history:
            chat_history = chat_history[-settings.rag_max_len_chat_history:]

        print("Risposta:")
        print(response["answer"])

if __name__ == "__main__":
    main()

import os

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

def main():
    dir_db = "data/chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(dir_db) and os.listdir(dir_db):
        print("Chroma database already exists. Skipping creation.")
        vectorstore = Chroma(persist_directory=dir_db, embedding_function=embeddings)
    else:
        print("Loading the PDF file...")
        reader = PdfReader("data/document.pdf")
        docs = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text:
                doc = Document(page_content=text, metadata={"page": i})
                docs.append(doc)
        print("Splitting the text...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        print("Creating vectorial database...")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="data/chroma_db")
    
    print("Searching...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("Configuring the LLM...")
    llm = OllamaLLM(model="qwen2.5:3b")

    system_prompt = (
        "Sei un assistente utile ed esperto. Usa SOLO il seguente contesto per rispondere alla domanda. "
        "Se non conosci la risposta in base al contesto, dì semplicemente che non lo sai, non inventare nulla.\n\n"
        "Contesto: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"), 
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("=" * 50)
    print("\nSISTEMA PRONTO!\n")
    print("Fai le tue domande sul PDF. Digita 'esci' per chiudere il programma.")
    print("=" * 50)
    while True:
        domanda = input("Domanda: ")
        if domanda == "esci":
            break

        print("Generazione della risposta in corso (potrebbe richiedere qualche secondo)...\n")

        # Interroghiamo il RAG
        response = rag_chain.invoke({"input": domanda})

        print("Risposta:")
        print(response["answer"])

if __name__ == "__main__":
    main()

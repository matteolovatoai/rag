from fastapi import FastAPI, HTTPException
from langchain_chroma import Chroma
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.schema import AIMessage, Document, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from contextlib import asynccontextmanager
from schemas.chat import ChatRequest, ChatResponse
from langchain_community.chat_message_histories import RedisChatMessageHistory

from pypdf import PdfReader

from config import settings

ml_models = {}

def init_rag_engine():
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    embeddings = OllamaEmbeddings(model=settings.embeddings_model)
    if settings.db_path.exists() and any(settings.db_path.iterdir()):
        vectorstore = Chroma(persist_directory=str(settings.db_path), embedding_function=embeddings)
    else:
        if not settings.pdf_path.exists():
            raise HTTPException(status_code=400, detail="PDF file not found")
        reader = PdfReader(str(settings.pdf_path))
        docs = [
            Document(page_content=page.extract_text(),metadata={"page": i})
            for i, page in enumerate(reader.pages, 1)
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=str(settings.db_path))

    llm = ChatOllama(model="qwen2.5:3b")
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.search_kwargs})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", settings.contextualize_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", settings.contextualize_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"), 
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    ml_models["rag_chain"] = create_retrieval_chain(history_aware_retriever, question_answer_chain)

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_rag_engine()
    yield
    ml_models.clear()

app = FastAPI(title="RAG", lifespan=lifespan)

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if "rag_chain" not in ml_models:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    chain = ml_models["rag_chain"]
    session_id = request.session_id

    chat_history = RedisChatMessageHistory(
        session_id=session_id,
        url=settings.redis_url
    )

    response = chain.invoke({"input": request.query, "chat_history": chat_history.messages})
    chat_history.add_user_message(request.query)
    chat_history.add_ai_message(response["answer"])

    if len(chat_history.messages) > settings.max_history * 2:
        messages_to_keep = chat_history.messages[-settings.max_history * 2:]
        chat_history.clear()
        for msg in messages_to_keep:
            chat_history.add_message(msg)

    return ChatResponse(answer=response["answer"])

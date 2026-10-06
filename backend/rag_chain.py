import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # type: ignore

load_dotenv()

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

def get_rag_chain():
    """Initializes the history-aware conversational RAG chain."""
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError("Vector database not found. Run ingestion first.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    # 1. Prompt to contextualize the user's latest question using history
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, formulate a standalone question. Do NOT answer it, just reformulate it if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # 2. Prompt to answer the contextualized question
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question using the context below.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 3. Combine both into a final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
    return rag_chain

def ask_question(question: str, chat_history: list) -> str:
    """Formats history and invokes the chain."""
    # Map Streamlit roles to LangChain roles
    formatted_history = []
    for msg in chat_history:
        if msg["role"] == "user":
            formatted_history.append(("human", msg["content"]))
        elif msg["role"] == "assistant" and "Hello!" not in msg["content"]:
            formatted_history.append(("ai", msg["content"]))

    chain = get_rag_chain()
    response = chain.invoke({"input": question, "chat_history": formatted_history})
    return response["answer"]
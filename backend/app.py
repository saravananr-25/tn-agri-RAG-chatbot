from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.ingestion import ingest_documents
from backend.rag_chain import ask_question

app = FastAPI(title="RAG Chatbot Backend")

# Update your Pydantic model
class QueryRequest(BaseModel):
    question: str
    chat_history: list = []  # Add this line


@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running."}


@app.post("/api/ingest")
def trigger_ingestion():
    """Endpoint to trigger the PDF ingestion pipeline."""
    try:
        result = ingest_documents()
        return result
    except FileNotFoundError as fnf_err:
        raise HTTPException(status_code=404, detail=str(fnf_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/chat")
def chat_endpoint(payload: QueryRequest):
    """Endpoint to handle user queries via RAG."""
    try:
        if not payload.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
        answer = ask_question(payload.question, payload.chat_history)
        return {"question": payload.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
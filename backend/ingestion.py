import os
from glob import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Define paths
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")


def ingest_documents(chunk_size: int = 1000, chunk_overlap: int = 200) -> dict:
    """Loads PDF files from data/, chunks them, and stores embeddings in ChromaDB."""
    pdf_files = glob(os.path.join(DATA_PATH, "*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {os.path.abspath(DATA_PATH)}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        all_docs.extend(loader.load())

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(all_docs)
    print(f"Split {len(all_docs)} pages into {len(chunks)} chunks.")

    # Generate embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return {
        "status": "success",
        "total_files": len(pdf_files),
        "total_pages": len(all_docs),
        "total_chunks": len(chunks),
        "persist_directory": os.path.abspath(CHROMA_PATH),
    }


if __name__ == "__main__":
    result = ingest_documents()
    print("Ingestion completed successfully:", result)
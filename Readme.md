# TN Agri RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to process and query agricultural documents, specifically referencing the `agri_e_pn_2024_25.pdf` data. The application features a decoupled architecture with a FastAPI backend for handling the RAG pipeline and a Streamlit frontend for the user interface.

## 🛠️ Tech Stack

*   **Frontend:** Streamlit
*   **Backend:** FastAPI, Uvicorn, Pydantic
*   **AI/LLM Framework:** LangChain (with OpenAI integrations)
*   **Vector Database:** ChromaDB
*   **Document Processing:** PyPDF

## 📂 Project Structure

```text
tn_agri_RAG_chatbot/
├── backend/                  # Backend API and RAG logic
│   ├── app.py                # FastAPI application entry point
│   ├── ingestion.py          # Script to process PDFs and load into ChromaDB
│   └── rag_chain.py          # LangChain RAG pipeline setup
├── chroma_db/                # Local ChromaDB vector storage (Auto-generated)
│   └── chroma.sqlite3
├── data/                     # Source documents
│   └── agri_e_pn_2024_25.pdf # Example agricultural policy/data document
├── frontend/                 # User interface
│   └── strlitapp.py          # Streamlit application
├── .env                      # Environment variables (API Keys) - Create this!
├── .gitignore
└── Requirements.txt          # Python dependencies
```

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd tn_agri_RAG_chatbot
```

### 2. Set up a virtual environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r Requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory and add your necessary API keys (e.g., OpenAI).
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Data Ingestion
Before running the app, process the PDF in the `data/` folder to populate your local ChromaDB vector store.
```bash
python backend/ingestion.py
```

### 6. Run the Application

You will need two terminal windows/tabs to run the backend and frontend simultaneously.

**Terminal 1 (Backend):**
Start the FastAPI server.
```bash
cd backend
uvicorn app:app --reload
```
*(The API will typically be available at `http://localhost:8000`)*

**Terminal 2 (Frontend):**
Start the Streamlit interface.
```bash
cd frontend
streamlit run strlitapp.py
```
*(The UI will typically open in your browser at `http://localhost:8501`)*

## 📝 Notes
*   Ensure that the `requests` library in the frontend is correctly pointing to your local FastAPI server url (`http://localhost:8000/...`).
*   The `chroma_db` folder will be generated and updated automatically when you run `ingestion.py`. It is recommended to keep `chroma_db/` in your `.gitignore`.
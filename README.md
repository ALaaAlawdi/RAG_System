# RAG System

A Retrieval-Augmented Generation (RAG) API that ingests PDF/TXT documents and answers questions based on their content using OpenAI embeddings and GPT-4o-mini.

## Stack

- **FastAPI** — API framework
- **ChromaDB** — vector store (persistent, local)
- **LangChain** — document chunking and retrieval
- **OpenAI** — embeddings (`text-embedding-3-large`) + chat (`gpt-4o-mini`)
- **PyPDF2** — PDF parsing

## Project Structure

```
RAG_System/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── .env                             # your secrets (gitignored)
├── .env.example
├── uploads/                         # uploaded files saved here (auto-created)
├── chroma_db/                       # ChromaDB persistent storage (auto-created)
└── app/
    ├── core/
    │   ├── config.py                # loads OPENAI_API_KEY from .env
    │   └── logger.py                # colorlog + file logger
    ├── knowledge_center/
    │   ├── setup.py                 # embeddings, vector store, text splitter, file readers
    │   └── center.py                # KnowledgeCenter class
    ├── routers/
    │   └── rag.py                   # POST /upload, POST /ask
    └── schemas/
        └── rag.py                   # Pydantic request/response models
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
active .venv/Scripts/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=sk-...
```

### 3. Run the server

```bash
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`.  
Interactive docs at `http://localhost:8000/docs`.

---

## API Reference

### `POST /api/v1/upload`

Upload a PDF or TXT document. The file is chunked and stored in ChromaDB.

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | PDF or TXT file |

**Response**

```json
{
  "success": true,
  "message": "Successfully uploaded 42 chunks to base_center",
  "chunks_added": 42
}
```

---

### `POST /api/v1/ask`

Ask a question. The system retrieves the most relevant chunks from all uploaded documents and generates an answer.

**Request** — `application/json`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | The question to answer |
| `k` | int | `4` | Number of chunks to retrieve |
| `file_path` | string | `null` | Limit search to a specific file (e.g. `uploads/report.pdf`) |

```json
{
  "query": "What are the main findings?",
  "k": 4,
  "file_path": "uploads/report.pdf"
}
```

**Response**

```json
{
  "answer": "The main findings indicate that...",
  "sources": [
    {
      "text": "...relevant chunk text...",
      "metadata": {
        "source": "uploads/report.pdf",
        "page": 3,
        "chunk": 1
      }
    }
  ]
}
```

---

## Notes

- Supported file types: `.pdf`, `.txt`
- Chunks: 1000 characters with 100-character overlap
- All uploads persist across restarts (ChromaDB stores to disk)
- No authentication required
- Logs written to `app.log` and console

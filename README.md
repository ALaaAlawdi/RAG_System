# RAG System

A Retrieval-Augmented Generation (RAG) API that ingests PDF/TXT documents and answers questions based on their content using OpenAI embeddings and GPT-4o.

## Stack

- **FastAPI** — API framework
- **ChromaDB** — vector store (persistent, local)
- **LangChain** — document chunking and retrieval
- **OpenAI** — embeddings (`text-embedding-3-large`) + chat (`gpt-4o`)
- **Ollama** — local LLM inference (`hf.co/AlaaAlawdi/llama_finetune`)
- **PyPDF2** — PDF parsing

## Project Structure

```
RAG_System/
├── main.py                          # FastAPI entry point
├── pyproject.toml                   # project metadata and dependencies
├── uv.lock                          # locked dependency tree
├── requirements.txt                 # pinned deps (pip fallback)
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

**Using uv (recommended):**

```bash
uv sync
```

**Using pip (fallback):**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
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

### 3. (Optional) Set up the fine-tuned Ollama model

Only required if you want to use `"provider": "finetune"` in `/ask`.

1. **Install Ollama** — download from [ollama.com](https://ollama.com/download)

2. **Pull the fine-tuned model:**

```bash
ollama pull hf.co/AlaaAlawdi/llama_finetune
```

3. **Make sure Ollama is running** before starting the server:

```bash
ollama serve
```

> If you skip this step, the API still works normally with `"provider": "openai"`.

---

### 4. Run the server

**Using uv:**

```bash
uv run uvicorn main:app --reload
```

**Using pip venv:**

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
| `provider` | string | `"openai"` | LLM to use: `"openai"` (gpt-4o) or `"finetune"` (local Ollama fine-tuned model) |

```json
{
  "query": "What are the main findings?",
  "k": 4,
  "file_path": "uploads/report.pdf",
  "provider": "openai"
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

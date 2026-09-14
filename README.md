# RAG Complete Pipeline

A Python Retrieval-Augmented Generation project using LangChain, OpenAI models, and ChromaDB. It includes document ingestion, vector retrieval, agentic RAG, multi-vector retrieval, and several RAG pattern examples.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- ChromaDB running as an HTTP server
- PowerShell on Windows, or an equivalent shell

## Setup on Windows

Open PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then activate `.venv` again.

## Configure `.env`

Create a file named `.env` in the project root. The application loads it automatically with `python-dotenv`.

Example configuration:

```env
OPENAI_API_KEY=your_openai_api_key_here

CHROMA_HOST=localhost
CHROMA_PORT=8000

DATA_DIR=data
COLLECTION_NAME=rag_documents
CHUNK_SIZE=800
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
TOP_K=5

RESTRICTED_FILES=
INGESTION_REGISTRY=ingestion_registry.json

AGENT_MAX_ITERATIONS=5
AGENT_MAX_SEARCHES=6
AGENT_MAX_RESULTS=5
AGENT_MAX_CHUNK_CHARS=4000
AGENT_MAX_DISTANCE=0.6
```

### Environment variables

- `OPENAI_API_KEY`: OpenAI authentication key. Keep it secret.
- `CHROMA_HOST` and `CHROMA_PORT`: address of the ChromaDB HTTP server.
- `DATA_DIR`: local folder containing source documents.
- `COLLECTION_NAME`: Chroma collection used by standard ingestion and retrieval.
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: text-splitting settings.
- `EMBEDDING_MODEL`: OpenAI embedding model used for vectors.
- `CHAT_MODEL`: OpenAI chat model used for answers.
- `TOP_K`: number of chunks returned during retrieval.
- `RESTRICTED_FILES`: optional comma-separated filenames marked with restricted access metadata.
- `INGESTION_REGISTRY`: local JSON file that stores first-ingestion timestamps. It should remain local.
- `AGENT_MAX_ITERATIONS`, `AGENT_MAX_SEARCHES`, `AGENT_MAX_RESULTS`, and `AGENT_MAX_CHUNK_CHARS`: limits for agentic RAG.
- `AGENT_MAX_DISTANCE`: maximum Chroma distance accepted as relevant evidence.

### Keep `.env` private

Do not commit `.env`, API keys, private documents, or generated local databases. The repository ignores `.env`, `data/`, `chroma_db/`, `ingestion_registry.json`, virtual environments, and Python cache files. The example above uses a placeholder key only.

If an API key is ever committed, revoke it immediately and create a new key. Deleting the file later does not remove the key from Git history.

## Start ChromaDB

Open a second PowerShell terminal, activate the virtual environment, and run:

```powershell
.\.venv\Scripts\Activate.ps1
chroma run --path .\chroma_db --host localhost --port 8000
```

Keep this terminal running. The Python code connects using `CHROMA_HOST` and `CHROMA_PORT`.

## Add source documents

Place local documents under `data/`. Supported file types are:

- PDF (`.pdf`)
- Text (`.txt`)
- Markdown (`.md`)
- Word documents (`.docx`)

The `data/` directory is intentionally excluded from Git because it may contain employee, medical, finance, security, or other confidential information.

## Ingest documents

With ChromaDB running:

```powershell
python -m src.ingestion.ingest
```

This loads documents, adds metadata, splits them into chunks, creates embeddings, and stores the vectors in ChromaDB. It also creates the local `ingestion_registry.json` file.

The standard ingestion flow does not recreate an existing collection. To delete the collection before ingesting again:

```powershell
python -m src.scripts.delete_collection
python -m src.ingestion.ingest
```

## Run the RAG applications

Basic RAG command-line interface:

```powershell
python -m src.core.basic_rag
```

Bounded agentic RAG:

```powershell
python -m src.patterns.agentic_rag
```

The safer agentic implementation, when present in the checkout:

```powershell
python -m src.patterns.agentic_rag_safe
```

Type a question at the prompt. Type `exit` or `quit` to stop.

## Other examples

Pattern implementations are in `src/patterns/`. Examples include:

```powershell
python -m src.patterns.citation_aware_rag
python -m src.patterns.hybrid_search
python -m src.patterns.query_rewriting
python -m src.patterns.reranking
python -m src.patterns.routing_rag
```

Multi-vector retrieval:

```powershell
python -m src.multi_vector_retrieval.mvr_ingest
python -m src.multi_vector_retrieval.mvr_retrieval
```

Show access-level metadata stored in the collection:

```powershell
python -m src.security.check_access_levels
```

## Project structure

```text
src/core/                   Shared RAG and prompt utilities
src/ingestion/              Document loading and chunking
src/multi_vector_retrieval/ Multi-vector ingestion and retrieval
src/patterns/               RAG pattern examples
src/security/               Access-level utilities
requirements.txt            Python dependencies
.env                        Local configuration; never commit
```

## Troubleshooting

**Chroma connection refused:** start the ChromaDB command and verify `CHROMA_HOST` and `CHROMA_PORT`.

**No documents found:** check `DATA_DIR`, confirm the `data/` directory exists, and use a supported file extension.

**Collection already exists:** run `python -m src.scripts.delete_collection` before ingesting again.

**OpenAI authentication or quota error:** verify `OPENAI_API_KEY`, account quota, and model names in `.env`.

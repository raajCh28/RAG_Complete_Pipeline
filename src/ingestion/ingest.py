import os
import glob
import json
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from src.ingestion.employee_chunking import split_employee_records

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

RESTRICTED_FILES = {
    file.strip()
    for file in os.getenv("RESTRICTED_FILES", "").split(",")
    if file.strip()
}

# Stores the first ingestion date of every file.
INGESTION_REGISTRY = os.getenv(
    "INGESTION_REGISTRY",
    "ingestion_registry.json"
)

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
}


def load_ingestion_registry():
    if not os.path.exists(INGESTION_REGISTRY):
        return {}

    try:
        with open(INGESTION_REGISTRY, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        print(
            f"Warning: Could not read {INGESTION_REGISTRY}. "
            "Starting with an empty registry."
        )
        return {}


def save_ingestion_registry(registry):
    with open(INGESTION_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)


def get_first_ingested_at(source, registry):
    # Normalize the path so the same file is always identified consistently.
    normalized_source = os.path.normcase(
        os.path.abspath(source)
    )

    if normalized_source in registry:
        return registry[normalized_source]["first_ingested_at"]

    first_ingested_at = int(time.time())

    registry[normalized_source] = {
        "first_ingested_at": first_ingested_at
    }

    return first_ingested_at


def get_category(source, data_dir):
    # Uses the first folder under DATA_DIR as the document category.
    source_path = Path(source).resolve()
    data_path = Path(data_dir).resolve()

    try:
        relative_path = source_path.relative_to(data_path)
    except ValueError:
        return "unknown"

    if len(relative_path.parts) > 1:
        return relative_path.parts[0].lower()

    # Files directly inside DATA_DIR get their filename stem as category.
    return source_path.stem.lower()


def load_documents(data_dir: str):
    docs = []

    for path in glob.glob(
        os.path.join(data_dir, "**/*"),
        recursive=True
    ):
        if os.path.isdir(path):
            continue

        ext = Path(path).suffix.lower()
        loader_factory = LOADERS.get(ext)

        if loader_factory is None:
            continue

        docs.extend(loader_factory(path).load())

    return docs


def enrich_metadata(docs, registry):
    for doc in docs:
        source = doc.metadata.get("source", "")
        filename = Path(source).name

        doc.metadata["filename"] = filename
        doc.metadata["file_type"] = Path(source).suffix.lstrip(".").lower()
        doc.metadata["category"] = get_category(source, DATA_DIR)
        doc.metadata["first_ingested_at"] = get_first_ingested_at(source, registry)
        doc.metadata["access_level"] = (
            "restricted" if filename in RESTRICTED_FILES else "public"
        )

    return docs


def get_chroma_client():
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT")),
    )


def collection_exists(client, name) -> bool:
    try:
        client.get_collection(name)
        return True
    except Exception:
        return False


def main():
    chroma_client = get_chroma_client()

    if collection_exists(chroma_client, COLLECTION_NAME):
        print(
            f"Collection '{COLLECTION_NAME}' already exists — "
            "delete it and run ingest.py again."
        )
        return

    registry = load_ingestion_registry()

    docs = load_documents(DATA_DIR)

    if not docs:
        print(
            f"No documents found in ./{DATA_DIR}. "
            "Add .txt, .md, .pdf, or .docx files and rerun."
        )
        return

    docs = enrich_metadata(docs, registry)
    save_ingestion_registry(registry)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs = split_employee_records(docs)
    chunks = splitter.split_documents(docs)

    file_count = len(
        set(
            d.metadata.get("source", "unknown")
            for d in docs
        )
    )

    categories = sorted(
        set(
            d.metadata.get("category", "unknown")
            for d in chunks
        )
    )

    print(
        f"Loaded {file_count} file(s) as "
        f"{len(docs)} document/page object(s)."
    )
    print(f"Split into {len(chunks)} chunk(s).")
    print(f"Categories: {', '.join(categories)}")

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        ),
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    vectorstore.add_documents(chunks)

    print(
        f"Ingested {len(chunks)} chunks "
        f"into '{COLLECTION_NAME}'."
    )

    print(
        f"Ingestion registry: "
        f"{INGESTION_REGISTRY}"
    )


if __name__ == "__main__":
    main()
import os
import json
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from document_loader import load_documents


# ---------------------------------------------------------
# Environment variables
# ---------------------------------------------------------

load_dotenv()


DATA_DIR = os.getenv(
    "DATA_DIR",
    "data"
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "new_documents"
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        800
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        100
    )
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small"
)

INGESTION_REGISTRY = os.getenv(
    "INGESTION_REGISTRY",
    "ingestion_registry.json"
)


# ---------------------------------------------------------
# Ingestion registry
# ---------------------------------------------------------

def load_ingestion_registry():

    if not os.path.exists(
        INGESTION_REGISTRY
    ):
        return {}

    try:

        with open(
            INGESTION_REGISTRY,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except (
        json.JSONDecodeError,
        OSError
    ):

        print(
            f"Warning: Could not read "
            f"{INGESTION_REGISTRY}. "
            f"Starting with an empty registry."
        )

        return {}


def save_ingestion_registry(registry):

    with open(
        INGESTION_REGISTRY,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            registry,
            f,
            indent=4
        )


# ---------------------------------------------------------
# First ingestion timestamp
# ---------------------------------------------------------

def get_first_ingested_at(
    source,
    registry
):

    normalized_source = os.path.normcase(
        os.path.abspath(source)
    )

    if normalized_source in registry:

        return registry[
            normalized_source
        ]["first_ingested_at"]

    first_ingested_at = int(
        time.time()
    )

    registry[
        normalized_source
    ] = {
        "first_ingested_at":
            first_ingested_at
    }

    return first_ingested_at


# ---------------------------------------------------------
# Metadata enrichment
# ---------------------------------------------------------

def enrich_metadata(
    docs,
    registry
):

    for doc in docs:

        source = doc.metadata.get(
            "source",
            ""
        )

        doc.metadata["filename"] = (
            Path(source).name
        )

        doc.metadata["file_type"] = (
            Path(source)
            .suffix
            .lstrip(".")
            .lower()
        )

        doc.metadata[
            "first_ingested_at"
        ] = get_first_ingested_at(
            source,
            registry
        )

    return docs


# ---------------------------------------------------------
# Chroma client
# ---------------------------------------------------------

def get_chroma_client():

    return chromadb.HttpClient(
        host=os.getenv(
            "CHROMA_HOST",
            "172.31.32.85"
        ),
        port=int(
            os.getenv(
                "CHROMA_PORT",
                8000
            )
        ),
    )


# ---------------------------------------------------------
# Check collection
# ---------------------------------------------------------

def collection_exists(
    client,
    name
):

    try:

        client.get_collection(
            name
        )

        return True

    except Exception:

        return False


# ---------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------

def main():

    print(
        "Starting ingestion..."
    )

    # -----------------------------------------------------
    # Connect to Chroma
    # -----------------------------------------------------

    chroma_client = (
        get_chroma_client()
    )

    # -----------------------------------------------------
    # Prevent accidental duplicate ingestion
    # -----------------------------------------------------

    if collection_exists(
        chroma_client,
        COLLECTION_NAME
    ):

        print(
            f"Collection "
            f"'{COLLECTION_NAME}' "
            f"already exists."
        )

        print(
            "Delete the collection "
            "and run ingest.py again."
        )

        return

    # -----------------------------------------------------
    # Load registry
    # -----------------------------------------------------

    registry = (
        load_ingestion_registry()
    )

    # -----------------------------------------------------
    # Load documents
    # -----------------------------------------------------

    docs = load_documents(
        DATA_DIR
    )

    if not docs:

        print(
            f"No documents found in "
            f"./{DATA_DIR}."
        )

        print(
            "Supported formats: "
            ".txt, .md, .pdf, .docx"
        )

        return

    # -----------------------------------------------------
    # Add metadata
    # -----------------------------------------------------

    docs = enrich_metadata(
        docs,
        registry
    )

    save_ingestion_registry(
        registry
    )

    # -----------------------------------------------------
    # Separate normal text and tables
    # -----------------------------------------------------

    normal_docs = [
        doc
        for doc in docs
        if doc.metadata.get(
            "content_type"
        ) != "table"
    ]

    table_docs = [
        doc
        for doc in docs
        if doc.metadata.get(
            "content_type"
        ) == "table"
    ]

    # -----------------------------------------------------
    # Create built-in text splitter
    # -----------------------------------------------------

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    )

    # -----------------------------------------------------
    # Split normal documents
    # -----------------------------------------------------

    normal_chunks = (
        splitter.split_documents(
            normal_docs
        )
    )

    # -----------------------------------------------------
    # Handle table rows
    #
    # Normally one row = one Document.
    #
    # If a row is larger than CHUNK_SIZE,
    # RecursiveCharacterTextSplitter is used.
    # -----------------------------------------------------

    table_chunks = []

    for table_doc in table_docs:

        if len(
            table_doc.page_content
        ) <= CHUNK_SIZE:

            table_chunks.append(
                table_doc
            )

        else:

            split_rows = (
                splitter.split_documents(
                    [table_doc]
                )
            )

            # Preserve table metadata
            for chunk in split_rows:

                chunk.metadata[
                    "content_type"
                ] = "table"

                chunk.metadata[
                    "table_index"
                ] = table_doc.metadata.get(
                    "table_index"
                )

                chunk.metadata[
                    "row_index"
                ] = table_doc.metadata.get(
                    "row_index"
                )

                table_chunks.append(
                    chunk
                )

    # -----------------------------------------------------
    # Combine normal chunks + table chunks
    # -----------------------------------------------------

    chunks = (
        normal_chunks +
        table_chunks
    )

    if not chunks:

        print(
            "No chunks were created."
        )

        return

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    file_count = len(
        set(
            d.metadata.get(
                "source",
                "unknown"
            )
            for d in docs
        )
    )

    print(
        f"Loaded {file_count} file(s)."
    )

    print(
        f"Loaded {len(docs)} "
        f"document/page/table-row objects."
    )

    print(
        f"Normal documents: "
        f"{len(normal_docs)}"
    )

    print(
        f"Table rows: "
        f"{len(table_docs)}"
    )

    print(
        f"Normal text chunks: "
        f"{len(normal_chunks)}"
    )

    print(
        f"Final chunks: "
        f"{len(chunks)}"
    )

    # -----------------------------------------------------
    # Create Chroma vector store
    # -----------------------------------------------------

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    vectorstore.add_documents(
        chunks
    )

    print(
        f"Successfully ingested "
        f"{len(chunks)} chunks into "
        f"'{COLLECTION_NAME}'."
    )

    print(
        f"Ingestion registry: "
        f"{INGESTION_REGISTRY}"
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
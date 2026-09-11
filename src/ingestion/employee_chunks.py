import os

import chromadb
from dotenv import load_dotenv


load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def get_collection():
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT")),
    )
    return client.get_collection(COLLECTION_NAME)


def get_employee_chunks():
    """Return only employee-record chunks from the Chroma collection."""
    collection = get_collection()
    data = collection.get(
        where={"entity_type": "employee"},
        include=["documents", "metadatas"],
    )

    chunks = []
    for chunk_id, document, metadata in zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    ):
        chunks.append({
            "id": chunk_id,
            "employee_id": metadata.get("employee_id", "unknown"),
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "unknown"),
            "document": document,
        })

    return sorted(chunks, key=lambda chunk: chunk["employee_id"])


def main():
    chunks = get_employee_chunks()

    if not chunks:
        print(
            "No employee chunks found. Rebuild the collection with "
            "employee_chunking.py through ingest.py first."
        )
        return

    print(f"Employee chunks: {len(chunks)}\n")

    for number, chunk in enumerate(chunks, start=1):
        print(f"#{number} | Employee ID: {chunk['employee_id']}")
        print(f"Source: {chunk['source']} | Page: {chunk['page']}")
        print(f"Chroma ID: {chunk['id']}")
        print(chunk["document"])
        print("-" * 70)


if __name__ == "__main__":
    main()
import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "new_documents")

client = chromadb.HttpClient(
    host=os.getenv("CHROMA_HOST", "172.31.32.85"),
    port=int(os.getenv("CHROMA_PORT", 8000)),
)


def main():
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        return

    data = collection.get(include=["documents", "metadatas"])
    total = len(data["ids"])
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks: {total}\n")

    for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        source = meta.get("source", "unknown")
        page = meta.get("page")
        label = f"{source} (page {page})" if page is not None else source

        print(f"ID: {id_}")
        print(f"Source: {label}")
        print(f"Length: {len(doc)} chars")
        print(f"Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print("-" * 60)


if __name__ == "__main__":
    main()
import os

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "new_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def get_vectorstore():
    # Connects to the existing Chroma collection.
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "172.31.32.85"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )

    return Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )


def show_access_levels():
    # Shows each file and its stored access level.
    vectorstore = get_vectorstore()
    data = vectorstore._collection.get(include=["metadatas"])

    file_access = {}

    for metadata in data["metadatas"]:
        filename = metadata.get("filename", "unknown")
        access_level = metadata.get("access_level", "unknown")
        file_access[filename] = access_level

    print("\nFile Access Levels")
    print("-" * 40)

    for filename, access_level in sorted(file_access.items()):
        print(f"{filename:<30} {access_level}")


if __name__ == "__main__":
    show_access_levels()
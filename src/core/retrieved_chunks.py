import os

from dotenv import load_dotenv
import chromadb

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

TOP_K = int(os.getenv("TOP_K"))

def get_chroma_client():
    return chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT")),
    )


def get_vectorstore():
    chroma_client = get_chroma_client()

    return Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(
            model=EMBEDDING_MODEL
        ),
    )


def get_available_categories():
    # Gets all categories currently stored in Chroma.
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)

    data = collection.get(
        include=["metadatas"]
    )

    categories = {
        metadata.get("category")
        for metadata in data.get("metadatas", [])
        if metadata and metadata.get("category")
    }

    return sorted(categories)


def retrieve_with_scores(query: str, top_k: int = TOP_K):
    # Returns [(document, distance), ...].
    vectorstore = get_vectorstore()

    return vectorstore.similarity_search_with_score(
        query,
        k=top_k
    )
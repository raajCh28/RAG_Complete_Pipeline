import os
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "new_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", 4))


def get_vectorstore():
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "172.31.32.85"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )
    return Chroma(
        client=chroma_client,
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
    )


def retrieve_with_scores(query: str, top_k: int = TOP_K):
    """Returns [(doc, distance), ...]. Pure retrieval — no printing, no side effects."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=top_k)
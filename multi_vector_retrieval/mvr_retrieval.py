import os
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

COLLECTION_NAME = "multi_vector_collection"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", 5))

embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

PROMPT = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""")


# Connects to the MVR collection in Chroma.
def get_collection():
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST"),
        port=int(os.getenv("CHROMA_PORT"))
    )
    return client.get_collection(COLLECTION_NAME)


# Searches the original, summary, and question vectors.
def retrieve(query, top_k=TOP_K):
    collection = get_collection()
    result = collection.query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=top_k
    )

    chunks = {}

    for doc, metadata, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0]
    ):
        chunk_id = metadata["chunk_id"]

        if chunk_id not in chunks:
            chunks[chunk_id] = {
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page", -1),
                "distance": distance,
                "matched_as": metadata.get("representation"),
                "original": None
            }

        if metadata.get("representation") == "original":
            chunks[chunk_id]["original"] = doc

        if distance < chunks[chunk_id]["distance"]:
            chunks[chunk_id]["distance"] = distance
            chunks[chunk_id]["matched_as"] = metadata.get("representation")

    # Fetches the original text for chunks matched through summary/questions.
    for chunk_id, chunk in chunks.items():
        if chunk["original"] is None:
            original = collection.get(
                where={"chunk_id": chunk_id}
            )

            for doc, metadata in zip(
                original["documents"],
                original["metadatas"]
            ):
                if metadata.get("representation") == "original":
                    chunk["original"] = doc
                    break

    return sorted(chunks.values(), key=lambda x: x["distance"])


# Builds the original document context for the LLM.
def build_context(chunks):
    return "\n\n---\n\n".join(
        f"Source: {c['source']}\nPage: {c['page']}\n{c['original']}"
        for c in chunks
        if c["original"]
    )


# Retrieves MVR chunks and generates the final answer.
def ask(query):
    chunks = retrieve(query)

    if not chunks:
        return "No relevant documents found.", []

    print("\nRetrieved chunks:")
    for i, c in enumerate(chunks, 1):
        print(
            f"{i}. {c['source']} | Page: {c['page']} | "
            f"Matched: {c['matched_as']} | Distance: {c['distance']:.4f}"
        )

    answer = (PROMPT | llm | StrOutputParser()).invoke({
        "context": build_context(chunks),
        "question": query
    })

    sources = sorted(set(c["source"] for c in chunks))
    return answer, sources


if __name__ == "__main__":
    print("Multi-Vector RAG Query Tool. Type 'exit' to quit.\n")

    while True:
        query = input("Question: ").strip()

        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer, sources = ask(query)
        print(f"\nAnswer: {answer}")

        if sources:
            print(f"Sources: {', '.join(sources)}")

        print()
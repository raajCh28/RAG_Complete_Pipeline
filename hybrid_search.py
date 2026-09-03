from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import PROMPT, CHAT_MODEL, build_context

load_dotenv()


# ---------------------------------------------------------
# BM25 INDEX
# ---------------------------------------------------------

def build_bm25_index(vectorstore):
    """
    Build a BM25 keyword-search index from the chunks
    already stored in ChromaDB.
    """

    data = vectorstore.get(include=["documents", "metadatas"])

    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        return None, [], []

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    return bm25, documents, metadatas


# ---------------------------------------------------------
# BM25 SEARCH
# ---------------------------------------------------------

def bm25_search(query, bm25, documents, metadatas, top_k=TOP_K):
    """
    Perform keyword-based BM25 search.
    """

    if bm25 is None:
        return []

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for index in ranked_indexes:
        results.append(
            {
                "content": documents[index],
                "metadata": metadatas[index],
                "score": scores[index],
                "index": index
            }
        )

    return results


# ---------------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------------

def vector_search(query, vectorstore, top_k=TOP_K):
    """
    Perform semantic/vector search using ChromaDB.
    """

    return vectorstore.similarity_search_with_score(
        query,
        k=top_k
    )


# ---------------------------------------------------------
# CHUNK IDENTIFIER
# ---------------------------------------------------------

def chunk_key_from_document(doc):
    """
    Creates a stable identifier for matching chunks
    between vector search and BM25 search.
    """

    return (
        doc.metadata.get("source", "unknown"),
        doc.metadata.get("page"),
        doc.page_content[:100]
    )


def chunk_key_from_bm25(result):
    """
    Creates the same type of identifier for BM25 results.
    """

    return (
        result["metadata"].get("source", "unknown"),
        result["metadata"].get("page"),
        result["content"][:100]
    )


# ---------------------------------------------------------
# RRF
# ---------------------------------------------------------

def reciprocal_rank_fusion(
    vector_results,
    bm25_results,
    top_k=TOP_K,
    rrf_k=60
):
    """
    Combine vector and BM25 rankings using
    Reciprocal Rank Fusion (RRF).

    RRF formula:

        score = 1 / (rrf_k + rank)
    """

    fused_scores = {}
    chunk_data = {}

    # -------------------------
    # Vector results
    # -------------------------

    for rank, (doc, distance) in enumerate(vector_results, start=1):

        key = chunk_key_from_document(doc)

        fused_scores[key] = fused_scores.get(key, 0) + (
            1 / (rrf_k + rank)
        )

        chunk_data[key] = doc

    # -------------------------
    # BM25 results
    # -------------------------

    for rank, result in enumerate(bm25_results, start=1):

        key = chunk_key_from_bm25(result)

        fused_scores[key] = fused_scores.get(key, 0) + (
            1 / (rrf_k + rank)
        )

        if key not in chunk_data:
            chunk_data[key] = result

    # -------------------------
    # Sort by RRF score
    # -------------------------

    ranked_results = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for key, rrf_score in ranked_results[:top_k]:

        data = chunk_data[key]

        if hasattr(data, "page_content"):
            # Vector Document
            doc = data

        else:
            # BM25 result
            from langchain_core.documents import Document

            doc = Document(
                page_content=data["content"],
                metadata=data["metadata"]
            )

        final_results.append(
            (doc, rrf_score)
        )

    return final_results


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

def print_vector_results(results):
    print("\n" + "=" * 70)
    print("VECTOR SEARCH RESULTS")
    print("=" * 70)

    for rank, (doc, distance) in enumerate(results, start=1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")

        print(f"\nRank {rank}")
        print(f"Source   : {source}")
        print(f"Page     : {page}")
        print(f"Distance : {distance:.4f}")
        print(f"Content  : {doc.page_content[:200]}...")


def print_bm25_results(results):
    print("\n" + "=" * 70)
    print("BM25 KEYWORD SEARCH RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        source = result["metadata"].get("source", "unknown")
        page = result["metadata"].get("page", "N/A")

        print(f"\nRank {rank}")
        print(f"Source : {source}")
        print(f"Page   : {page}")
        print(f"Score  : {result['score']:.4f}")
        print(f"Content: {result['content'][:200]}...")


def print_hybrid_results(results):
    print("\n" + "=" * 70)
    print("HYBRID SEARCH RESULTS (RRF)")
    print("=" * 70)

    for rank, (doc, rrf_score) in enumerate(results, start=1):

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")

        print(f"\nRank {rank}")
        print(f"Source    : {source}")
        print(f"Page      : {page}")
        print(f"RRF Score : {rrf_score:.6f}")
        print(f"Content   : {doc.page_content[:200]}...")


# ---------------------------------------------------------
# HYBRID RETRIEVAL
# ---------------------------------------------------------

def hybrid_retrieve(query, top_k=TOP_K):

    vectorstore = get_vectorstore()

    # Build BM25 from existing Chroma documents
    bm25, documents, metadatas = build_bm25_index(vectorstore)

    if bm25 is None:
        return [], [], []

    # -------------------------
    # Vector search
    # -------------------------

    vector_results = vector_search(
        query,
        vectorstore,
        top_k
    )

    # -------------------------
    # BM25 search
    # -------------------------

    bm25_results = bm25_search(
        query,
        bm25,
        documents,
        metadatas,
        top_k
    )

    # -------------------------
    # Hybrid / RRF
    # -------------------------

    hybrid_results = reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        top_k
    )

    return (
        vector_results,
        bm25_results,
        hybrid_results
    )


# ---------------------------------------------------------
# RAG ASK
# ---------------------------------------------------------

def ask(query):

    vector_results, bm25_results, hybrid_results = hybrid_retrieve(
        query
    )

    if not hybrid_results:
        return "No documents found — check ingest.py has run.", []

    # Display retrieval comparison
    print(f"\nOriginal query: {query}")

    print_vector_results(vector_results)
    print_bm25_results(bm25_results)
    print_hybrid_results(hybrid_results)

    # Use hybrid results for final answer
    context = build_context(hybrid_results)

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2
    )

    chain = PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    sources = sorted(
        set(
            doc.metadata.get("source", "unknown")
            for doc, _ in hybrid_results
        )
    )

    return answer, sources


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("RAG Hybrid Search. Type 'exit' to quit.\n")

    while True:

        query = input("Question: ").strip()

        if query.lower() in ("exit", "quit"):
            break

        if not query:
            continue

        answer, sources = ask(query)

        print("\n" + "=" * 70)
        print("FINAL ANSWER")
        print("=" * 70)

        print(answer)

        if sources:
            print(
                f"\nSources: {', '.join(sources)}"
            )

        print()
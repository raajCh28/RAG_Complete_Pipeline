from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import PROMPT, CHAT_MODEL, build_context

load_dotenv()

# Retrieve more candidates first, then rerank them.
RETRIEVAL_K = 10

# Number of chunks finally passed to the answer-generation LLM.
FINAL_K = TOP_K


RERANK_PROMPT = ChatPromptTemplate.from_template(
    """
You are a document relevance evaluator.

Your task is to determine how relevant the following document
chunk is to the user's question.

Give a relevance score from 0 to 10:

10 = Directly and completely answers the question
8-9 = Highly relevant and contains most of the required information
6-7 = Relevant but only partially answers the question
4-5 = Somewhat related but does not answer the question well
2-3 = Weakly related
0-1 = Not relevant

IMPORTANT:
- Judge relevance to the question, not writing quality.
- Do not use information outside the provided chunk.
- Return ONLY the numeric score.
- Do not provide an explanation.

User Question:
{question}

Document Chunk:
{document}

Relevance Score:
"""
)


# ---------------------------------------------------------
# RERANKER
# ---------------------------------------------------------

def get_reranker():
    """
    Creates the OpenAI model used for reranking. Not necessarily the CHAT_MODEL.
    """

    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )


def calculate_relevance_score(query, document, llm):
    """
    Ask model to score one document chunk
    against the user's query.
    """

    chain = RERANK_PROMPT | llm | StrOutputParser()

    response = chain.invoke(
        {
            "question": query,
            "document": document
        }
    ).strip()

    try:
        score = float(response)

        # Keep score within the expected 0-10 range.
        score = max(0.0, min(10.0, score))

        return score

    except ValueError:
        print(
            f"Warning: Could not parse reranker score: {response}"
        )
        return 0.0


# ---------------------------------------------------------
# INITIAL VECTOR RETRIEVAL
# ---------------------------------------------------------

def retrieve_candidates(query, retrieval_k=RETRIEVAL_K):
    """
    First retrieve candidate chunks using Chroma
    semantic/vector search.

    The reranker does NOT search the entire database.
    It only evaluates these candidates.
    """

    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search_with_score(
        query,
        k=retrieval_k
    )

    return results


# ---------------------------------------------------------
# RERANK
# ---------------------------------------------------------

def rerank_documents(query, candidates, final_k=FINAL_K):
    """
    Rerank the retrieved candidates using the reranker model.
    """

    llm = get_reranker()

    reranked = []

    for original_rank, (doc, vector_distance) in enumerate(
        candidates,
        start=1
    ):

        relevance_score = calculate_relevance_score(
            query,
            doc.page_content,
            llm
        )

        reranked.append(
            {
                "doc": doc,
                "vector_distance": vector_distance,
                "original_rank": original_rank,
                "reranker_score": relevance_score
            }
        )

    # Sort by LLM relevance score.
    reranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True
    )

    return reranked[:final_k]


# ---------------------------------------------------------
# DISPLAY INITIAL RETRIEVAL
# ---------------------------------------------------------

def print_initial_results(results):

    print("\n" + "=" * 75)
    print("INITIAL VECTOR SEARCH RESULTS")
    print("=" * 75)

    for rank, (doc, distance) in enumerate(results, start=1):

        source = doc.metadata.get("source", "unknown")

        page = doc.metadata.get("page", "N/A")

        print(f"\nRank              : {rank}")
        print(f"Source            : {source}")
        print(f"Page              : {page}")
        print(f"Vector Distance   : {distance:.4f}")
        print(
            f"Content           : "
            f"{doc.page_content[:250]}..."
        )


# ---------------------------------------------------------
# DISPLAY RERANKED RESULTS
# ---------------------------------------------------------

def print_reranked_results(results):

    print("\n" + "=" * 75)
    print("RERANKED RESULTS")
    print("=" * 75)

    for new_rank, item in enumerate(results, start=1):

        doc = item["doc"]

        source = doc.metadata.get("source", "unknown")

        page = doc.metadata.get("page", "N/A")

        print(f"\nNew Rank          : {new_rank}")
        print(f"Original Rank     : {item['original_rank']}")
        print(f"Source            : {source}")
        print(f"Page              : {page}")
        print(
            f"Vector Distance   : "
            f"{item['vector_distance']:.4f}"
        )
        print(
            f"Reranker Score    : "
            f"{item['reranker_score']:.2f}/10"
        )
        print(
            f"Content           : "
            f"{doc.page_content[:250]}..."
        )


# ---------------------------------------------------------
# RAG ASK
# ---------------------------------------------------------

def ask(query):
    # -----------------------------------------------------
    # STEP 1: VECTOR RETRIEVAL
    # -----------------------------------------------------
    candidates = retrieve_candidates(query, RETRIEVAL_K)

    if not candidates:
        return (
            "No documents found — check ingest.py has run.",
            []
        )

    print_initial_results(candidates)

    # -----------------------------------------------------
    # STEP 2: RERANK
    # -----------------------------------------------------

    reranked = rerank_documents(query, candidates, FINAL_K)

    print_reranked_results(reranked)

    # -----------------------------------------------------
    # STEP 3: BUILD CONTEXT
    # -----------------------------------------------------

    final_documents = [
        (item["doc"], item["reranker_score"])
        for item in reranked
    ]

    context = build_context(final_documents)

    # -----------------------------------------------------
    # STEP 4: FINAL ANSWER
    # -----------------------------------------------------

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
            item["doc"].metadata.get(
                "source",
                "unknown"
            )
            for item in reranked
        )
    )

    return answer, sources


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        "RAG Reranking Pipeline. "
        "Type 'exit' to quit.\n"
    )

    while True:

        query = input("Question: ").strip()

        if query.lower() in (
            "exit",
            "quit"
        ):
            break

        if not query:
            continue

        answer, sources = ask(query)

        print("\n" + "=" * 75)
        print("FINAL ANSWER")
        print("=" * 75)

        print(answer)

        if sources:
            print(
                f"\nSources: "
                f"{', '.join(sources)}"
            )

        print()

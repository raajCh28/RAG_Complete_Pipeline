"""
    Complete Guarded RAG pipeline.
    Flow:
        Input Guard
             ↓
        Domain Guard
             ↓
        Retrieval Guard
             ↓
          ChromaDB
             ↓
        Output Guard
             ↓
           LLM
             ↓
          Answer
"""

import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import PROMPT, CHAT_MODEL, print_retrieved, build_context

load_dotenv()


# ============================================================
# INPUT GUARD
# ============================================================

# Patterns that indicate prompt injection or attempts to
# override the assistant's instructions.
BLOCKED_PATTERNS = [
    r"ignore (all|any|previous|the) instructions",
    r"reveal (your|the) system prompt",
    r"you are now",
    r"disregard (all|any|previous) rules",
    r"pretend (you are|to be)",
]


def input_guard(query: str) -> str | None:
    """
    Checks the raw user question for common prompt-injection
    patterns.

    Returns:
        Rejection reason if blocked.
        None if the question is allowed.
    """

    lowered = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return (
                f"Blocked: question matched a restricted "
                f"pattern ('{pattern}')."
            )

    return None


# ============================================================
# DOMAIN GUARD
# ============================================================

DOMAIN_DESCRIPTION = """
The knowledge base belongs to a company called Test_HCL.

The available company information covers only these areas:

1. Employee information
2. Leave policies
3. Finance and expenses
4. IT and security
5. Medical benefits

A question should be considered IN-DOMAIN if its meaning is
related to any of these Test_HCL areas.

The wording does not need to contain exact keywords from these
categories. Users may ask questions informally, indirectly,
or conversationally.

A question should be considered OUT-OF-DOMAIN if it is unrelated
to Test_HCL company information, policies, employees, or the
areas listed above.

Return only one word:

ALLOW

or

BLOCK
"""


def domain_guard(query: str) -> str | None:
    """
    Uses an LLM as a semantic classifier to determine whether
    the question belongs to the Test_HCL knowledge domain.

    Returns:
        None if the question is allowed.
        A rejection reason if the question is outside the domain.
    """

    guard_llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

    guard_prompt = f"""
You are a strict domain classifier for a company knowledge-base
assistant.

Determine whether the user's question belongs to the allowed
Test_HCL knowledge domain.

{DOMAIN_DESCRIPTION}

User question:
{query}

Return only:
ALLOW
or
BLOCK
"""

    try:
        result = guard_llm.invoke(guard_prompt)

        decision = result.content.strip().upper()

        # Remove possible extra whitespace or formatting.
        decision = decision.replace("`", "").strip()

        if decision == "ALLOW":
            return None

        if decision == "BLOCK":
            return (
                "Blocked: question is outside the Test_HCL "
                "knowledge domain."
            )

        # Fail closed if the classifier returns something unexpected.
        return (
            "Blocked: domain guard could not safely classify "
            "the question."
        )

    except Exception as e:
        print(f"Domain guard error: {e}")

        # Fail closed for safety.
        return (
            "Blocked: domain guard could not process "
            "the question safely."
        )


# ============================================================
# RETRIEVAL GUARD
# ============================================================

RESTRICTED_FILES = {
    "medical.docx"
}


def retrieval_guard_filter():
    """
    Creates a Chroma metadata filter that attempts to exclude
    restricted documents at the database level.

    This is a best-effort filter only.
    enforce_output_guard() below is the actual enforcement layer.
    """

    if not RESTRICTED_FILES:
        return None

    return {
        "filename": {
            "$nin": list(RESTRICTED_FILES)
        }
    }


# ============================================================
# OUTPUT GUARD
# ============================================================

def enforce_output_guard(results):
    """
    Explicitly removes restricted documents from retrieved
    results.

    This provides a second enforcement layer independent of
    the Chroma metadata filter.
    """

    safe_results = []
    blocked_count = 0

    for doc, score in results:

        filename = doc.metadata.get("filename", "")

        if filename in RESTRICTED_FILES:
            blocked_count += 1
            continue

        safe_results.append((doc, score))

    if blocked_count:
        print(
            f"Output guard: stripped "
            f"{blocked_count} restricted chunk(s) "
            f"from the results."
        )

    return safe_results


# ============================================================
# MAIN GUARDED RAG PIPELINE
# ============================================================

def ask(query: str, top_k: int = TOP_K):

    # 1. INPUT GUARD
    block_reason = input_guard(query)

    if block_reason:
        print(block_reason)

        return (
            "This question can't be processed as written.",
            []
        )


    # 2. DOMAIN GUARD
    domain_block_reason = domain_guard(query)

    if domain_block_reason:
        print(domain_block_reason)

        return (
            "I can only answer questions related to "
            "Test_HCL employees, leave, finance, IT and security",
            []
        )


    # 3. RETRIEVAL GUARD
    vectorstore = get_vectorstore()
    where = retrieval_guard_filter()

    # Request a small buffer above TOP_K so that restricted
    # chunks removed by the output guard do not unnecessarily
    # reduce the final number of safe results.

    buffer_k = top_k + len(RESTRICTED_FILES) * 2

    results = vectorstore.similarity_search_with_score(
        query,
        k=buffer_k,
        filter=where
    )


    # 4. OUTPUT GUARD
    results = enforce_output_guard(results)

    # Keep only the requested number of safe results.
    results = results[:top_k]

    if not results:
        return (
            "No safe documents were found for this question.",
            []
        )

    print_retrieved(query, results)
    context = build_context(results)

    # GENERATION
    llm = ChatOpenAI( model=CHAT_MODEL,temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})


    # COLLECT SOURCES
    sources = sorted(
        set(
            doc.metadata.get(
                "filename",
                doc.metadata.get(
                    "source",
                    "unknown"
                )
            )
            for doc, _ in results
        )
    )
    return answer, sources


if __name__ == "__main__":
    print("Guarded RAG. Type 'exit' to quit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer, sources = ask(q)
        print(f"\nAnswer: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()
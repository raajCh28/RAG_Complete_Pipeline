import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import PROMPT, CHAT_MODEL, print_retrieved, build_context

load_dotenv()

# --- INPUT GUARD CONFIG ---
# Regex patterns that indicate a prompt-injection attempt or a request to
# break the assistant's instructions. Checked BEFORE any retrieval or LLM
# call, so a blocked question never costs an API call and never enters the
# model's context window at all.
BLOCKED_PATTERNS = [
    r"ignore (all|any|previous|the) instructions",
    r"reveal (your|the) system prompt",
    r"you are now",
    r"disregard (all|any|previous) rules",
    r"pretend (you are|to be)",
]

# --- RETRIEVAL GUARD CONFIG ---
# Filenames excluded from retrieval because their content is sensitive
# (medical.docx contains health information). Keyed on the `filename`
# metadata field ingest.py already stores — no ingest.py changes needed.
RESTRICTED_FILES = {"medical.docx"}


def input_guard(query: str) -> str | None:
    """Checks the raw question against known bad patterns before doing
    anything else. Returns a rejection reason if blocked, or None if the
    question is allowed through. Simple regex matching on purpose — not a
    full ML classifier, but catches the common injection phrasings."""
    lowered = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return f"Blocked: question matched a restricted pattern ('{pattern}')."
    return None


def retrieval_guard_filter():
    """Best-effort pre-filter sent to Chroma, excluding restricted filenames
    at the database level. NOT trusted as the only enforcement layer —
    Chroma's $nin operator has a known history of silently returning
    everything instead of actually filtering (chroma-core/chroma issue
    #1424), so relying on this alone for something security-relevant would
    be a real risk. enforce_output_guard() below is the actual guarantee;
    this filter is still worth sending since it reduces noise when it does
    work correctly, and costs nothing when it doesn't."""
    if not RESTRICTED_FILES:
        return None
    return {"filename": {"$nin": list(RESTRICTED_FILES)}}


def enforce_output_guard(results):
    """The real enforcement layer: explicitly strips any chunk from a
    restricted file out of the results in plain Python, regardless of
    whether the database-level filter above actually worked. Never trust a
    single point of enforcement for something security-sensitive."""
    safe_results = []
    blocked_count = 0
    for doc, score in results:
        filename = doc.metadata.get("filename", "")
        if filename in RESTRICTED_FILES:
            blocked_count += 1
            continue
        safe_results.append((doc, score))

    if blocked_count:
        print(f"Output guard: stripped {blocked_count} restricted chunk(s) from the results.")

    return safe_results


def ask(query: str, top_k: int = TOP_K):
    """Full guarded pipeline: input guard -> filtered retrieval -> output
    guard -> generation. Any block along the way returns early with a
    clear, honest reason instead of silently failing or half-answering."""

    # 1. INPUT GUARD
    block_reason = input_guard(query)
    if block_reason:
        print(block_reason)
        return "This question can't be processed as written.", []

    # 2. RETRIEVAL GUARD
    # Requesting extra buffer beyond top_k: if the database-level filter
    # above doesn't actually exclude anything (see the known-bug note),
    # restricted chunks could otherwise consume result slots that should
    # have gone to allowed content once enforce_output_guard() removes them.
    vectorstore = get_vectorstore()
    where = retrieval_guard_filter()
    buffer_k = top_k + len(RESTRICTED_FILES) * 2
    results = vectorstore.similarity_search_with_score(query, k=buffer_k, filter=where)

    # 3. OUTPUT GUARD — the layer actually being trusted
    results = enforce_output_guard(results)
    results = results[:top_k]  # trim back down to what was actually asked for

    if not results:
        return "No documents found — check ingest.py has run, or all matching content is restricted.", []

    print_retrieved(query, results)
    context = build_context(results)

    # 4. GENERATION — only reached once the question and results both passed guards
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = sorted(set(doc.metadata.get("filename", doc.metadata.get("source", "unknown")) for doc, _ in results))
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
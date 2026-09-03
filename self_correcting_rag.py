from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL, build_context, PROMPT

load_dotenv()

MAX_ATTEMPTS = 3

# Judges whether a draft answer is actually backed by the given context,
# separate from whatever call produced the draft, so grading isn't biased
# by the same reasoning that generated it.
GRADE_PROMPT = ChatPromptTemplate.from_template(
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Draft answer: {answer}\n\n"
    "Is the draft answer FULLY supported by the context above, with no "
    "unsupported claims or guesses? Respond with only YES or NO."
)

# Used only when the draft failed grading — broadens/rephrases the query
# to give retrieval a better shot on the next attempt.
REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "The following question could not be answered well from the documents "
    "retrieved so far. Rewrite it as a broader or differently phrased search "
    "query that might surface more relevant content. Output only the "
    "rewritten query, nothing else.\n\nOriginal question: {question}\n"
    "Previous search query: {search_query}\n\nNew search query:"
)


def retrieve(search_query: str, top_k: int):
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(search_query, k=top_k)


def generate_draft(original_question: str, context: str) -> str:
    """Always answers the ORIGINAL question, even though the search query
    used for retrieval may have been rewritten by this point."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": original_question})


def is_dont_know(draft: str) -> bool:
    """An 'I don't know' answer is trivially 'grounded' (it makes no
    unsupported claims), so a plain groundedness check would wrongly accept
    it as a success. Treat it as a failure worth retrying instead."""
    text = draft.lower()
    return "don't know" in text or "cannot answer" in text or "no information" in text


def grade_answer(original_question: str, context: str, draft_answer: str) -> bool:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = GRADE_PROMPT | llm | StrOutputParser()
    verdict = chain.invoke({
        "context": context,
        "question": original_question,
        "answer": draft_answer,
    }).strip().upper()
    return verdict.startswith("YES") and not is_dont_know(draft_answer)


def rewrite_search_query(original_question: str, previous_search_query: str) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "question": original_question,
        "search_query": previous_search_query,
    }).strip()


def ask(query: str, max_attempts: int = MAX_ATTEMPTS):
    search_query = query
    top_k = TOP_K
    draft = None
    last_results = []  # only updated when retrieval actually returns something,
                        # so it always stays in sync with whatever `draft` was based on

    for attempt in range(1, max_attempts + 1):
        print(f"\nAttempt {attempt}: searching for \"{search_query}\" (top_k={top_k})")

        results = retrieve(search_query, top_k)
        if not results:
            print("  No chunks retrieved.")
            if attempt < max_attempts:          # fix: don't rewrite on a final, wasted attempt
                search_query = rewrite_search_query(query, search_query)
                top_k += 2
            continue

        last_results = results                  # fix: keep draft and its source chunks paired
        context = build_context(results)
        draft = generate_draft(query, context)
        grounded = grade_answer(query, context, draft)
        print(f"  Grounded in context: {'YES' if grounded else 'NO'}")

        if grounded:
            sources = sorted(set(doc.metadata.get("filename", doc.metadata.get("source", "unknown")) for doc, _ in last_results))
            return draft, sources, attempt

        if attempt < max_attempts:
            search_query = rewrite_search_query(query, search_query)
            top_k += 2

    sources = sorted(set(doc.metadata.get("filename", doc.metadata.get("source", "unknown")) for doc, _ in last_results)) if last_results else []
    disclaimer = (
        "\n\n[Note: could not fully verify this answer against the retrieved "
        "documents after multiple attempts — treat it with caution.]"
    )
    return (draft or "I don't know.") + disclaimer, sources, max_attempts


if __name__ == "__main__":
    print("Self-correcting RAG. Type 'exit' to quit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer, sources, attempts_used = ask(q)
        print(f"\nAnswer ({attempts_used} attempt(s)): {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()
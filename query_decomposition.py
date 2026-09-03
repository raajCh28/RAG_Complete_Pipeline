import json

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL, build_context

load_dotenv()

DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    "Break the following question into 1 to 4 simpler, standalone sub-questions "
    "that together cover everything being asked. Each sub-question must focus on "
    "exactly ONE distinct topic — never merge two unrelated topics into a single "
    "sub-question, even if the original question runs them together with no clear "
    "separator like 'and'.\n\n"
    "If the question is already a single, atomic topic, return it unchanged as the "
    "only item.\n\n"
    "Example:\n"
    "Question: what is leave policy time limit for expense claim\n"
    "Sub-questions: [\"What is the leave policy?\", \"What is the time limit for submitting an expense claim?\"]\n\n"
    "Respond with ONLY a JSON list of strings, nothing else.\n\n"
    "Question: {question}"
)

# NEW: dedicated answer prompt for this file (no longer reuses rag.py's PROMPT).
# Two changes from the shared PROMPT that fix the "I don't know" bug:
# 1. Explicitly allows answering multiple questions separately.
# 2. Only says "don't know" about the SPECIFIC part that's missing,
#    instead of refusing the whole answer because of one gap.
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question(s) below using only the context provided. "
    "If there is more than one question, address each one separately and clearly. "
    "If a specific part cannot be answered from the context, say so for that part "
    "only — do not refuse the whole answer just because one part is missing.\n\n"
    "Context:\n{context}\n\nQuestion(s):\n{question}\n\nAnswer:"
)


def decompose_query(query: str) -> list[str]:
    """Sends the original question to the LLM and gets back a list of
    simpler sub-questions. Falls back to [query] unchanged if the LLM's
    response isn't valid JSON, so a parsing hiccup can't break the pipeline."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = DECOMPOSE_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": query}).strip()

    try:
        sub_questions = json.loads(raw)
        if isinstance(sub_questions, list) and sub_questions:
            return sub_questions
    except json.JSONDecodeError:
        pass

    return [query]


def retrieve_for_subquestions(sub_questions: list[str], top_k: int = TOP_K):
    """Runs a separate similarity search per sub-question, merges results,
    deduplicating by chunk identity and keeping the best (lowest) distance seen."""
    vectorstore = get_vectorstore()
    seen = {}

    for sub_q in sub_questions:
        results = vectorstore.similarity_search_with_score(sub_q, k=top_k)
        for doc, score in results:
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])
            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    return sorted(seen.values(), key=lambda pair: pair[1])


def build_answer_question(query: str, sub_questions: list[str]) -> str:
    """if there's only one sub-question, nothing to clarify — use the original as-is. If there
    are multiple, send ONLY the clean numbered sub-questions as 'the question'."""
    if len(sub_questions) <= 1:
        return query
    return "\n".join(f"{i}. {sq}" for i, sq in enumerate(sub_questions, start=1))


def print_decomposition(query: str, sub_questions: list[str], results):
    """Pure display — no retrieval or LLM calls happen here."""
    print(f"\nOriginal question: {query}")
    print("Sub-questions:")
    for i, sub_q in enumerate(sub_questions, start=1):
        print(f"  {i}. {sub_q}")

    print(f"\nMerged top {len(results)} chunk(s) across all sub-questions:\n")
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source} (page {page})" if page is not None else source
        print(f"#{rank} | Distance: {score:.4f} | {label}")
        print(doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""))
        print("-" * 60)


def ask(query: str):
    sub_questions = decompose_query(query)
    results = retrieve_for_subquestions(sub_questions)

    if not results:
        return "No documents found — check ingest.py has run.", []

    print_decomposition(query, sub_questions, results)

    context = build_context(results)
    answer_question = build_answer_question(query, sub_questions)  # CHANGED

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = ANSWER_PROMPT | llm | StrOutputParser()  # CHANGED: uses ANSWER_PROMPT, not rag.py's PROMPT
    answer = chain.invoke({"context": context, "question": answer_question})

    sources = sorted(set(doc.metadata.get("source", "unknown") for doc, _ in results))
    return answer, sources


if __name__ == "__main__":
    print("RAG query tool with query decomposition. Type 'exit' to quit.\n")
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
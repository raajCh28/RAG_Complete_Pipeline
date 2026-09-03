from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL, build_context, print_retrieved

load_dotenv()

MAX_HISTORY_TURNS = 5

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    "Determine whether the current question depends on the conversation history.\n\n"
    "A question is a follow-up only if understanding it requires information "
    "from a previous turn, such as a pronoun, vague reference, omitted subject, "
    "or continuation of the previous topic.\n\n"
    "If it is a follow-up, rewrite it as a fully standalone question.\n"
    "If it is a new independent question, return it unchanged.\n\n"
    "Conversation history:\n{history}\n\n"
    "Current question:\n{question}\n\n"
    "Return exactly two lines:\n"
    "FOLLOW_UP: YES or NO\n"
    "QUESTION: <standalone question>"
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are answering questions using only the context below. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Conversation so far:\n{history}\n\n"
    "Context:\n{context}\n\n"
    "Current question: {question}\n\nAnswer:"
)


def format_history(chat_history: list) -> str:
    if not chat_history:
        return "(no previous turns yet)"
    lines = []
    for q, a in chat_history:
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a}")
    return "\n".join(lines)


def analyze_question(question: str, chat_history: list):
    if not chat_history:
        return False, question

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    chain = CONDENSE_PROMPT | llm | StrOutputParser()

    response = chain.invoke({
        "history": format_history(chat_history),
        "question": question,
    }).strip()

    lines = response.splitlines()

    follow_up = False
    standalone_question = question

    for line in lines:
        line = line.strip()

        if line.upper().startswith("FOLLOW_UP:"):
            value = line.split(":", 1)[1].strip().upper()
            follow_up = value == "YES"

        elif line.upper().startswith("QUESTION:"):
            standalone_question = line.split(":", 1)[1].strip()

    # Safety: an independent question must never be rewritten.
    if not follow_up:
        standalone_question = question

    return follow_up, standalone_question


def build_fallback_query(question: str, chat_history: list, is_follow_up: bool) -> str | None:

    if not is_follow_up or not chat_history:
        return None

    last_question, _ = chat_history[-1]

    return f"{last_question} {question}"


def retrieve_with_fallback(standalone_question: str, fallback_query: str | None, top_k: int = TOP_K):
    vectorstore = get_vectorstore()

    seen = {}

    queries = [standalone_question]

    if fallback_query:
        queries.append(fallback_query)

    for query in queries:

        for doc, score in vectorstore.similarity_search_with_score(query, k=top_k):

            key = (
                doc.metadata.get("source"),
                doc.metadata.get("page"),
                doc.page_content[:50]
            )

            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    return sorted(seen.values(), key=lambda pair: pair[1])[:top_k]


def ask(question: str, chat_history: list):

    is_follow_up, standalone_question = analyze_question(question, chat_history)

    if is_follow_up:
        print(f"Follow-up detected: YES")
        print(f"Standalone question: {standalone_question}")
    else:
        print(f"Follow-up detected: NO")
        print(f"Standalone question: {question}")

    fallback_query = build_fallback_query(question, chat_history, is_follow_up)

    results = retrieve_with_fallback(standalone_question, fallback_query)

    if not results:
        return (
            "No documents found — check ingest.py has run.",
            [],
            chat_history
        )

    print_retrieved(standalone_question, results)

    context = build_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    answer = chain.invoke({
        "history": format_history(chat_history),
        "context": context,
        "question": question,
    })

    sources = sorted(
        set(
            doc.metadata.get("source", "unknown")
            for doc, _ in results
        )
    )

    updated_history = chat_history + [
        (question, answer)
    ]

    updated_history = updated_history[-MAX_HISTORY_TURNS:]

    return answer, sources, updated_history


if __name__ == "__main__":
    print("Conversational RAG. Type 'exit' to quit.\n")
    chat_history = []

    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer, sources, chat_history = ask(q, chat_history)
        print(f"\nAnswer: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()
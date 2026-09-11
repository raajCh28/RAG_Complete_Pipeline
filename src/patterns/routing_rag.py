from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.core.basic_rag import ask as basic_ask, CHAT_MODEL
from src.core.prompts import load_prompt
from src.patterns.retrieval_augmented_summarization import summarize as summarize_topic

load_dotenv()

ROUTER_PROMPT = load_prompt("router")


# The only valid outputs the router should ever produce. Used to validate the LLM's response.
VALID_LABELS = {"qa", "summarize", "unsupported"}


def route(query: str) -> str:
    """Runs the ONE classification call for this question. temperature=0
    since classification should be consistent, not creative — the same
    question should route the same way every time.

    Falls back to 'qa' (the safest, most general path) if the model's
    response doesn't match one of the three expected labels, so a
    malformed classification can't crash the pipeline or silently do
    nothing."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = ROUTER_PROMPT | llm | StrOutputParser()
    label = chain.invoke({"question": query}).strip().lower()
    return label if label in VALID_LABELS else "qa"


def ask(query: str):
    """The actual routing step: dispatches to a completely different
    pipeline depending on the classification, and returns whatever that
    pipeline returns. This function does NOT retrieve anything or call an
    answer-generation LLM itself — that work belongs entirely to whichever
    branch gets chosen below."""
    label = route(query)
    print(f"Routed as: {label}")

    if label == "unsupported":
        # No retrieval, no generation call at all.
        return (
            "This looks like a counting or aggregation question across many "
            "individual records. This RAG pipeline retrieves and reasons over "
            "text passages — it can't reliably count or compute exact totals "
            "across structured records that way. An exact answer needs a "
            "different approach: loading the data into a structured format "
            "(e.g. a table) and querying it directly with code."
        ), []

    if label == "summarize":
        return summarize_topic(query)

    return basic_ask(query)


if __name__ == "__main__":
    print("Routing RAG. Type 'exit' to quit.\n")
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
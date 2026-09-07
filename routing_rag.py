from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Reusing the ALREADY BUILT pipelines directly — their prompts (PROMPT inside
# rag.py, SUMMARY_PROMPT inside retrieval_augmented_summarization.py) stay
# exactly where they are. This file adds only ONE new prompt: the router's
# own classification prompt, since nothing else in the project does that job.
from basic_rag import ask as basic_ask, CHAT_MODEL
from retrieval_augmented_summarization import summarize as summarize_topic

load_dotenv()

# This prompt's only job is classification — deciding
# WHERE a question should go, not answering it. This runs BEFORE any
# retrieval happens, which is the defining trait of routing: the decision is
# made up front, then only one path actually executes.

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Classify the user's query into exactly ONE of these categories:\n\n"

    "qa - A focused request for specific information, facts, rules, "
    "requirements, explanations, instructions, procedures, processes, "
    "conditions, or details about a particular subject. The answer may "
    "contain multiple points or steps, but the request remains focused "
    "on a specific subject.\n\n"

    "summarize - A request for a broad overview, summary, synthesis, or "
    "general understanding of a topic, policy, document, or related set "
    "of information. The request requires combining and condensing "
    "information rather than answering one focused question.\n\n"

    "unsupported - A request that cannot be reliably handled by any of "
    "the available pipelines or capabilities of this system. This includes "
    "requests requiring capabilities, information sources, data, or "
    "operations that are not available to the current system.\n\n"

    "CLASSIFICATION RULES:\n"
    "1. Classify according to the user's intended task and scope, not "
    "individual keywords.\n"
    "2. A focused question remains qa even if its answer requires multiple "
    "facts, steps, or passages.\n"
    "3. Use summarize only when the requested scope is broad and requires "
    "synthesis or an overall view.\n"
    "4. Use unsupported only when the requested task itself cannot be "
    "handled by the available system capabilities.\n"
    "5. Do not classify a query as unsupported merely because the required "
    "information may not exist in the knowledge base. Let retrieval and "
    "validation determine whether sufficient supporting information exists.\n"
    "6. Return exactly one category: qa, summarize, or unsupported.\n\n"

    "Query: {question}"
)


# The only valid outputs the router should ever produce. Used to validate
# the LLM's response, since it's plain text and could technically return
# something malformed or unexpected.
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
        # No retrieval, no generation call at all — declining is a decision
        # made from the classification alone. This is intentional: sending
        # this kind of question through an LLM would likely produce a
        # confident-sounding but wrong number, which is worse than an
        # honest refusal.
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
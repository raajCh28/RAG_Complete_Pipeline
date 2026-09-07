from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore
from basic_rag import CHAT_MODEL, print_retrieved, build_context

load_dotenv()

# summarization's goal is coverage across the whole collection, not precision on a single best match.
BROAD_TOP_K = 12

# detail_level is plain text injected into the prompt (e.g. "brief, 3-4
# sentence" or "detailed, comprehensive") — lets length/depth be controlled
# per call with no code changes, matching the "control length & detail"
# idea from the original pattern description.
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "First, determine whether the context below is actually related to the "
    "given topic. If NONE of the context is meaningfully related to the "
    "topic, respond with exactly: \"No relevant content found for this topic "
    "in the available documents.\" Do not force a connection, and do not "
    "summarize unrelated content as though it addresses the topic.\n\n"
    "If the context IS related, write a {detail_level} summary synthesizing "
    "a coherent overview across ALL the relevant context below — do not just "
    "answer a single question, and do not simply list chunks one by one. "
    "Group related points together even if they come from different sources.\n\n"
    "Context:\n{context}\n\nTopic: {topic}\n\nSummary:"
)


def retrieve_broad(topic: str, top_k: int = BROAD_TOP_K):
    """Retrieves a WIDE set of chunks relevant to the topic across the whole
    collection — no file restriction. This is the key structural difference
    from every other pattern file: coverage over precision, so every
    document touching the topic gets a chance to contribute."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(topic, k=top_k)


def summarize(topic: str, detail_level: str = "detailed, comprehensive"):
    """Named summarize(), not ask() like the other pattern files — this
    isn't answering a question, it's synthesizing an overview, so the
    function name reflects the different task shape."""
    results = retrieve_broad(topic)

    if not results:
        return "No documents found — check ingest.py has run.", []

    # print_retrieved(topic, results)
    context = build_context(results)  # reused as-is from basic_rag.py — same "Source: X" block format works fine here too

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)  # slightly higher than QA's 0.2 — synthesis benefits from a bit more natural phrasing than strict fact-copying
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    summary = chain.invoke({"context": context, "topic": topic, "detail_level": detail_level})

    sources = sorted(set(doc.metadata.get("filename", doc.metadata.get("source", "unknown")) for doc, _ in results))
    return summary, sources


if __name__ == "__main__":
    print("Retrieval-augmented summarization. Type 'exit' to quit.")
    print("Enter a TOPIC (not a specific question) to get a synthesized summary.\n")
    while True:
        topic = input("Topic: ").strip()
        if topic.lower() in ("exit", "quit"):
            break
        if not topic:
            continue

        detail = input("Detail level (brief/detailed, Enter for detailed): ").strip().lower()
        detail_level = "brief, 3-4 sentence" if detail == "brief" else "detailed, comprehensive"

        summary, sources = summarize(topic, detail_level)
        print(f"\nSummary:\n{summary}")
        if sources:
            print(f"\nSources: {', '.join(sources)}")
        print()
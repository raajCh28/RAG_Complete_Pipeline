import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.retrieved_chunks import get_vectorstore
from src.core.basic_rag import CHAT_MODEL, print_retrieved, build_context
from src.core.prompts import load_prompt


load_dotenv()

# summarization's goal is coverage across the whole collection, not precision on a single best match.
BROAD_TOP_K = int(os.getenv("BROAD_TOP_K"))

SUMMARY_PROMPT = load_prompt("summary")

def retrieve_broad(topic: str, top_k: int = BROAD_TOP_K):
    """Retrieves a WIDE set of chunks relevant to the topic across the whole collection."""
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
    context = build_context(results)  # reused as-is from basic_rag.py

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
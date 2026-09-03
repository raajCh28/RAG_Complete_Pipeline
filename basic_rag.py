import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import retrieve_with_scores

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the context below. "
    "If the answer isn't in the context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def print_retrieved(query: str, results):
    print(f"\nTop {len(results)} retrieved chunks for: \"{query}\"\n")
    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source} (page {page})" if page is not None else source

        print(f"#{rank} | Distance: {score:.4f} | {label}")
        print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        print("-" * 60)


def build_context(results) -> str:
    blocks = []
    for doc, _ in results:
        label = doc.metadata.get("source", "unknown")
        if "page" in doc.metadata:
            label += f" (page {doc.metadata['page']})"
        blocks.append(f"Source: {label}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def ask(query: str):
    results = retrieve_with_scores(query)
    print_retrieved(query, results)

    context = build_context(results)
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = sorted(set(doc.metadata.get("source", "unknown") for doc, _ in results))
    return answer, sources


if __name__ == "__main__":
    print("RAG query tool. Type 'exit' to quit.\n")
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
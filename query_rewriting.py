from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import PROMPT, CHAT_MODEL, print_retrieved, build_context

load_dotenv()

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the following user question into a clear, standalone search query "
    "suitable for retrieving relevant document chunks. Keep it concise and factual. "
    "Output only the rewritten query, nothing else.\n\n"
    "Question: {question}\n\nRewritten query:"
)


def rewrite_query(query: str) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query}).strip()


def retrieve_with_rewrite(query: str, top_k: int = TOP_K):
    rewritten = rewrite_query(query)
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(rewritten, k=top_k)
    return results, rewritten


def ask(query: str):
    results, rewritten = retrieve_with_rewrite(query)
    print(f"Rewritten query: {rewritten}")

    if not results:
        return "No documents found — check ingest.py has run.", []

    print_retrieved(query, results)
    context = build_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})  # original wording, not rewritten

    sources = sorted(set(doc.metadata.get("source", "unknown") for doc, _ in results))
    return answer, sources


if __name__ == "__main__":
    print("RAG query tool with query rewriting. Type 'exit' to quit.\n")
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
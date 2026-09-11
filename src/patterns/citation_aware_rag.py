import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.core.retrieved_chunks import get_vectorstore
from src.core.prompts import load_prompt

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
CITATION_PROMPT = load_prompt("citation_answer")


def build_numbered_context(results):
    """Assigns each retrieved chunk a reference number [1], [2], ... and builds
    both the numbered context string for the prompt AND a lookup dict mapping
    each number back to its source/page/snippet — used to print a references
    list after the answer and to validate the citations the model actually used."""
    blocks = []
    references = {}

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page")
        label = f"{source} (page {page})" if page is not None else source

        blocks.append(f"[{i}] Source: {label}\n{doc.page_content}")
        references[i] = {"label": label, "snippet": doc.page_content[:250]}

    return "\n\n---\n\n".join(blocks), references


def find_cited_numbers(answer: str) -> set:
    """Extracts every [N] the model actually used in its answer text, via regex."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def check_citations(answer: str, references: dict):
    """Raises if the model invents a citation that was never provided.
    Also warns about chunks that were retrieved but never cited."""
    cited = find_cited_numbers(answer)
    valid = set(references.keys())

    invalid = cited - valid
    unused = valid - cited

    if invalid:
        raise ValueError(
            f"Hallucinated citation(s): {sorted(invalid)}. "
            "The answer referenced context numbers that were not retrieved."
        )

    if unused:
        print(f"Note: chunk(s) {sorted(unused)} were retrieved but never cited in the answer.")

    return True


def print_references(references: dict, results: list | None = None):
    print("\nReferences:")
    for i, ref in references.items():
        distance = results[i - 1][1] if results and i - 1 < len(results) else "?"
        print(f"\n  [{i}] {ref['label']} | Distance: {distance}")
        print(f"      \"{ref['snippet']}...\"")
        print("      " + "-" * 56)


def ask(query: str, top_k: int | None = None):
    if top_k is None:
        top_k = DEFAULT_TOP_K

    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        return "No documents found — check ingest.py has run.", {}

    context, references = build_numbered_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = CITATION_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    check_citations(answer, references)

    return answer, references, results


if __name__ == "__main__":
    print("Citation-aware RAG. Type 'exit' to quit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer, references, results = ask(q)
        print(f"\nAnswer: {answer}")
        if references:
            print_references(references, results)
        print()
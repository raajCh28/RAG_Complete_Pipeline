import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL

load_dotenv()

CITATION_PROMPT = ChatPromptTemplate.from_template(
    "Answer the question using only the numbered context blocks below. "
    "After every claim or sentence that relies on a specific block, add its "
    "reference number in square brackets immediately after it, e.g. [1] or [2]. "
    "If a sentence draws on multiple blocks, cite all of them, e.g. [1][3]. "
    "Only use reference numbers that actually appear below — never invent one. "
    "If the answer isn't in the context, say you don't know and cite nothing.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


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
    """Flags a citation number the model invented that was never actually
    provided (hallucinated citation), and separately notes chunks that were
    retrieved but never cited at all (may be fine — just worth a glance)."""
    cited = find_cited_numbers(answer)
    valid = set(references.keys())

    invalid = cited - valid
    unused = valid - cited

    if invalid:
        print(f"Warning: answer cites {sorted(invalid)} — not in the retrieved context (hallucinated citation).")
    if unused:
        print(f"Note: chunk(s) {sorted(unused)} were retrieved but never cited in the answer.")


def print_references(references: dict):
    print("\nReferences:")
    for i, ref in references.items():
        print(f"  [{i}] {ref['label']}")
        print(f"      \"{ref['snippet']}...\"")


def ask(query: str, top_k: int = TOP_K):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    if not results:
        return "No documents found — check ingest.py has run.", {}

    context, references = build_numbered_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = CITATION_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    check_citations(answer, references)

    return answer, references


if __name__ == "__main__":
    print("Citation-aware RAG. Type 'exit' to quit.\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        answer, references = ask(q)
        print(f"\nAnswer: {answer}")
        if references:
            print_references(references)
        print()
"""
Step-back RAG pipeline.
    Flow:

        Original Question
                ↓
        Step-Back Prompt (LLM)
        generates a broader/general question
                ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
  Step-Back Question              Original Question
        ↓                               ↓
     ChromaDB                        ChromaDB
   (broad retrieval)              (specific retrieval)
        ↓                               ↓
        └───────────────┬───────────────┘
                         ↓
              Merge + Deduplicate
        (keep lowest-distance version
         of each chunk found by either query)
                         ↓
                  Build Context
          (mix of broad + specific chunks)
                         ↓
              Answer Prompt (LLM)
     (explicitly told the context is a mix,
      different wording than the question ≠ "don't know")
                         ↓
                Final Answer
        (uses ORIGINAL question wording,
         not the step-back question) 
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL, print_retrieved, build_context

load_dotenv()

STEP_BACK_PROMPT = ChatPromptTemplate.from_template(
    "Given the following specific question, write a more general, higher-level "
    "question about the broader topic it belongs to. The general question should "
    "be answerable from background/overview content, not just the specific detail "
    "being asked about.\n\n"
    "Example:\n"
    "Specific question: What is the time limit to submit an expense claim after "
    "an emergency business trip?\n"
    "Step-back question: What is the company's expense claim policy?\n\n"
    "Specific question: {question}\n\nStep-back question:"
)

# Step-back retrieval deliberately mixes broad/background chunks with
# specific ones, so the answer prompt needs to explicitly tolerate that mix
# and be told that different wording between question and context is
# expected, not a reason to refuse.
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "The context below combines both general/background information and "
    "specific details relevant to the question — some chunks being broader "
    "in scope than others is expected. Read all of it and use whatever is "
    "relevant, even if the answer is stated using different wording than "
    "the question itself. Answer using only information actually present "
    "in the context — do not invent facts not stated there. Only say you "
    "don't know if the context genuinely contains nothing relevant, not "
    "merely because the wording differs from how the question was asked.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def generate_step_back_question(query: str) -> str:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = STEP_BACK_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query}).strip()


def chunk_key(doc):
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])


def retrieve_with_stepback(query: str, step_back_question: str, top_k: int = TOP_K):
    """Returns ALL unique chunks found by either query — not trimmed to
    top_k, so the correct chunk from either pass can't get squeezed out."""
    vectorstore = get_vectorstore()
    seen = {}

    for q in (step_back_question, query):
        for doc, score in vectorstore.similarity_search_with_score(q, k=top_k):
            key = chunk_key(doc)
            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    return sorted(seen.values(), key=lambda pair: pair[1])


def ask(query: str):
    step_back_question = generate_step_back_question(query)
    print(f"Step-back question: {step_back_question}")

    results = retrieve_with_stepback(query, step_back_question)

    if not results:
        return "No documents found — check ingest.py has run.", []

    print_retrieved(query, results)
    context = build_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = ANSWER_PROMPT | llm | StrOutputParser()  # dedicated prompt, not rag.py's PROMPT
    answer = chain.invoke({"context": context, "question": query})

    sources = sorted(set(doc.metadata.get("filename", doc.metadata.get("source", "unknown")) for doc, _ in results))
    return answer, sources


if __name__ == "__main__":
    print("Step-back RAG. Type 'exit' to quit.\n")
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
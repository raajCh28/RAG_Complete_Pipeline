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
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.core.retrieved_chunks import get_vectorstore, TOP_K
from src.core.basic_rag import CHAT_MODEL, print_retrieved, build_context
from src.core.prompts import load_prompt

load_dotenv()


# Defines whether Step-Back should be used and the broader question if needed.
class StepBackDecision(BaseModel):
    use_step_back: bool = Field(
        description="Whether the original question would benefit from Step-Back retrieval."
    )
    step_back_question: str = Field(
        description="A broader retrieval question, or an empty string when Step-Back is not needed."
    )


STEP_BACK_PROMPT = load_prompt("step_back")
ANSWER_PROMPT = load_prompt("step_back_answer")

def generate_step_back_decision(query: str) -> StepBackDecision:
    """Decides whether Step-Back is needed and generates the broader question."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    structured_llm = llm.with_structured_output(StepBackDecision)

    chain = STEP_BACK_PROMPT | structured_llm

    return chain.invoke({"question": query})


def chunk_key(doc):
    """Creates a unique key for a retrieved chunk."""
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.page_content[:50],
    )


def retrieve_direct(query: str, top_k: int = TOP_K):
    """Retrieves chunks using only the original question."""
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=top_k)


def retrieve_with_stepback(
    query: str,
    step_back_question: str,
    top_k: int = TOP_K,
):
    """Retrieves using both Step-Back and original questions and removes duplicates."""
    vectorstore = get_vectorstore()
    seen = {}

    for q in (step_back_question, query):
        results = vectorstore.similarity_search_with_score(q, k=top_k)

        for doc, score in results:
            key = chunk_key(doc)

            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

    return sorted(seen.values(), key=lambda pair: pair[1])


def ask(query: str):
    """Runs Step-Back RAG and returns the answer with its sources."""
    decision = generate_step_back_decision(query)

    print(f"\nUse Step-Back: {decision.use_step_back}")

    if decision.use_step_back:
        print(f"Step-Back question: {decision.step_back_question}")

        results = retrieve_with_stepback(
            query,
            decision.step_back_question,
            TOP_K,
        )
    else:
        print("Step-Back question: Not used")

        results = retrieve_direct(query, TOP_K)

    if not results:
        return "No documents found — check ingest.py has run.", []

    print_retrieved(query, results)

    context = build_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    answer = chain.invoke(
        {
            "context": context,
            "question": query,
        }
    )

    sources = sorted(
        set(
            doc.metadata.get(
                "filename",
                doc.metadata.get("source", "unknown"),
            )
            for doc, _ in results
        )
    )

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
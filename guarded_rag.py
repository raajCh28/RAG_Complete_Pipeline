"""
Guarded RAG pipeline.

Flow:

    User Query
         ↓
    Input Guard
         ↓
    Domain Guard
         ↓
    Access-Controlled Retrieval
         ↓
    Context Guard
         ↓
    Answerability Guard
         ↓
    Grounded LLM
         ↓
       Answer
"""

import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import print_retrieved, build_context

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")


# GUARD 1: INPUT GUARD
BLOCKED_PATTERNS = [
    r"reveal (your|the) system prompt",
    r"you are now",
    r"pretend (you are|to be)",
]


def input_guard(query: str) -> str | None:
    # Returns a reason when the query matches a blocked pattern.
    lowered = query.lower()

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered):
            return "Blocked: question matched a restricted pattern."

    return None


# GUARD 2: DOMAIN GUARD

# Determine whether the user's question belongs to any domain that
# exists in the current knowledge base.

DOMAIN_PROMPT = ChatPromptTemplate.from_template(
    """
You are a domain classifier for a RAG system.

Available knowledge-base categories:
{categories}

Determine whether the user's question belongs to at least one of
the available categories.

Return exactly one word:

YES
or
NO

Return YES when the question can reasonably be associated with one
or more of the available categories.

Return NO when the question does not belong to any available category.

Do not answer the question.
Do not use outside knowledge.
Only classify the question against the available categories.

Question:
{question}

Decision:
"""
)


def get_available_categories(vectorstore):
    # Reads all unique categories currently stored in Chroma.
    data = vectorstore._collection.get(include=["metadatas"])

    categories = set()

    for metadata in data.get("metadatas", []):
        category = metadata.get("category")

        if category:
            categories.add(str(category))

    return sorted(categories)


def domain_guard(query: str, vectorstore) -> bool:
    # Checks whether the question belongs to a category in the knowledge base.
    categories = get_available_categories(vectorstore)

    if not categories:
        return False

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

    chain = DOMAIN_PROMPT | llm | StrOutputParser()

    decision = chain.invoke({
        "categories": ", ".join(categories),
        "question": query
    })

    return decision.strip().upper() == "YES"


# GUARD 3: ACCESS CONTROL
# Prevent restricted documents from being retrieved.
# Retrieval is filtered before the documents can become LLM context.

def access_guard_filter():
    # Retrieves only documents allowed for the current application.
    return {"access_level": "public"}


# GUARD 4: CONTEXT GUARD
# Provide defense-in-depth for access control.
# The retrieval filter is the primary protection. This second check
# removes any chunk that is not explicitly public before context is
# sent to the LLM.

def enforce_context_guard(results):
    # Removes every chunk that is not explicitly marked as public.
    safe_results = []

    for doc, score in results:
        access_level = str(
            doc.metadata.get("access_level", "")
        ).lower()

        if access_level != "public":
            continue

        safe_results.append((doc, score))

    return safe_results


# GUARD 5: ANSWERABILITY GUARD

# Check whether the retrieved context actually contains enough information to answer the user's question.
# This guard therefore checks the relationship between the question and the retrieved context.

# ANSWERABILITY_PROMPT = ChatPromptTemplate.from_template(
#     """
# You are an answerability checker for a company RAG system.

# Determine whether the provided context contains enough information
# to answer the user's question.

# Return exactly one word:

# YES
# or
# NO

# Return YES only when the context contains information that directly
# supports answering the question.

# Return NO when:
# - the context is unrelated
# - the context is insufficient
# - the answer requires information that is not present
# - the context only partially supports the answer

# Do not use outside knowledge.

# Context:
# {context}

# Question:
# {question}

# Decision:
# """
# )


# def answerability_guard(query: str, context: str) -> bool:
#     # Returns True only when the retrieved context supports the question.
#     llm = ChatOpenAI(
#         model=CHAT_MODEL,
#         temperature=0
#     )

#     chain = ANSWERABILITY_PROMPT | llm | StrOutputParser()

#     decision = chain.invoke({
#         "context": context,
#         "question": query
#     })

#     return decision.strip().upper() == "YES"


ANSWERABILITY_DISTANCE_THRESHOLD = float(
    os.getenv("ANSWERABILITY_DISTANCE_THRESHOLD")
)


def answerability_guard(results) -> bool:
    # Checks whether the best retrieved chunk is close enough to the query.
    if not results:
        return False

    best_distance = min(
        score for _, score in results
    )

    return best_distance <= ANSWERABILITY_DISTANCE_THRESHOLD


# FINAL GROUNDED ANSWER

# Generate the final response only after all guards have passed.
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
Answer the question using only the provided company context.

Rules:
- Use only information contained in the context.
- Do not use outside knowledge.
- Do not invent or assume missing information.
- If the context does not contain the answer, say: "I can't find any relevant information related to this question."
- Give a concise and direct answer.

Context:
{context}

Question:
{question}

Answer:
"""
)


def generate_answer(query: str, context: str) -> str:
    # Generates the final answer using only approved context.
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.2
    )

    chain = ANSWER_PROMPT | llm | StrOutputParser()

    return chain.invoke({
        "context": context,
        "question": query
    })


# MAIN RAG PIPELINE

# Execute all guards in the correct order and stop the pipeline when
# a query fails a security, domain, access, or answerability check.

def ask(query: str, top_k: int = TOP_K):

    # Guard 1:
    # Block obvious prompt-injection attempts.
    block_reason = input_guard(query)

    if block_reason:
        print(block_reason)
        return (
            "This question can't be processed as written.",
            []
        )

    # Guard 2:
    # Reject questions that do not belong to any category currently
    # available in the knowledge base.
    vectorstore = get_vectorstore()

    if not domain_guard(query, vectorstore):
        return (
            "I can only answer questions related to the available "
            "company information.",
        []
    )

    # Guard 3:
    # Retrieve only documents allowed by the access policy.
    results = vectorstore.similarity_search_with_score(
        query,
        k=top_k,
        filter=access_guard_filter()
    )

    # Guard 4:
    # Remove anything that should not reach the LLM.
    results = enforce_context_guard(results)

    if not results:
        return (
            "I can't find any relevant information related to this question.",
            []
        )

    print_retrieved(query, results)
    context = build_context(results)

    # Guard 5:
    # Check whether the retrieved context actually supports an answer.
    if not answerability_guard(results):
        return (
            "I can't find any relevant information related to this question.",
            []
        )

    # Generate the final grounded answer.
    answer = generate_answer(query, context)

    # Return only sources that were allowed into the final context.
    sources = sorted(
        set(
            doc.metadata.get(
                "filename",
                doc.metadata.get("source", "unknown")
            )
            for doc, _ in results
        )
    )

    return answer, sources



if __name__ == "__main__":
    print("Guarded RAG. Type 'exit' to quit.\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer, sources = ask(query)
        print(f"\nAnswer: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()
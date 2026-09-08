from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from retrieved_chunks import get_vectorstore, TOP_K
from basic_rag import CHAT_MODEL

load_dotenv()

MAX_ITERATIONS = 5
STALL_LIMIT = 2  # consecutive iterations with NO new chunks found before giving up early

# Strengthened: explicitly forbids answering from general knowledge, and
# requires searching before answering. This is layer 1 (a request to the
# model) — layer 2, the actual enforcement, is finalize() below, which
# doesn't trust the model to have obeyed this.
SYSTEM_MESSAGE = SystemMessage(content=(
    "You answer questions ONLY using information found via the "
    "search_documents tool. You must use the tool at least once before "
    "answering any question — never answer directly from your own general "
    "knowledge, even if you think you know the answer. You may call the "
    "tool more than once, with different queries, if a single search does "
    "not cover everything the question is asking. Once you have enough "
    "information, respond with your final answer directly. If the "
    "documents do not contain relevant information for this question, say "
    "so honestly rather than guessing or using outside knowledge."
))


def chunk_key(doc):
    """Identity for a retrieved chunk — used to detect whether a new
    search is actually finding NEW information, at the chunk level rather
    than the filename level (so a question answerable entirely from
    different sections of ONE file isn't mistaken for a stalled loop)."""
    return (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:50])


def make_search_tool(sources_seen: set, chunks_seen: set):
    """Fresh tool per ask() call. Both sets are updated as a side effect
    whenever the tool actually runs — sources_seen for display, chunks_seen
    (finer-grained) for stall detection and the no-content refusal check."""

    @tool
    def search_documents(query: str) -> str:
        """Search the company's documents for information relevant to the
        given query. Use this whenever you need more information to answer
        the question. You can call this again with a different query if
        your first search didn't cover everything you need."""
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=TOP_K)

        if not results:
            print("    -> no chunks found")
            return "No relevant chunks found for this query."

        lines = []
        found_from = set()
        for doc, score in results:
            source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
            sources_seen.add(source)
            chunks_seen.add(chunk_key(doc))
            found_from.add(source)
            lines.append(f"[{source}] {doc.page_content}")

        print(f"    -> found {len(results)} chunk(s) from: {', '.join(sorted(found_from))}")
        return "\n\n".join(lines)

    return search_documents


def finalize(answer: str, sources_seen: set, chunks_seen: set):
    """The ACTUAL enforcement layer, not just a prompt request: if nothing
    relevant was ever genuinely retrieved across every search performed —
    whether because the topic isn't in the documents, or the model never
    searched at all — the LLM's answer is discarded entirely and replaced
    with a fixed refusal. This is what actually prevents an off-topic
    question from getting answered from general knowledge, regardless of
    what the model generated or how confidently it said it."""
    if not chunks_seen:
        return (
            "I can't answer this from the available documents — no relevant "
            "content was found for this question."
        ), []
    return answer, sorted(sources_seen)


def ask(query: str, max_iterations: int = MAX_ITERATIONS):
    sources_seen = set()
    chunks_seen = set()
    search_documents = make_search_tool(sources_seen, chunks_seen)
    llm_with_tools = ChatOpenAI(model=CHAT_MODEL, temperature=0).bind_tools([search_documents])

    messages = [SYSTEM_MESSAGE, HumanMessage(content=query)]
    llm_calls = 0
    searches_run = 0
    stall_count = 0
    previous_chunk_count = 0
    stopped_early_reason = None

    for iteration in range(1, max_iterations + 1):
        response = llm_with_tools.invoke(messages)
        llm_calls += 1
        messages.append(response)

        if not response.tool_calls:
            print(f"\n({llm_calls} LLM call(s) total, {searches_run} search(es) run)")
            return finalize(response.content, sources_seen, chunks_seen)

        for tool_call in response.tool_calls:
            print(f"  [Iteration {iteration}] Searching: \"{tool_call['args'].get('query')}\"")
            tool_result = search_documents.invoke(tool_call)
            messages.append(tool_result)
            searches_run += 1

        # Stall check: did this round of searches find any NEW chunks at all?
        if len(chunks_seen) == previous_chunk_count:
            stall_count += 1
        else:
            stall_count = 0
        previous_chunk_count = len(chunks_seen)

        if stall_count >= STALL_LIMIT:
            stopped_early_reason = f"no new information found for {stall_count} iterations in a row"
            print(f"  Stopping early — {stopped_early_reason}.")
            break

    reason = stopped_early_reason or f"reached the {max_iterations}-iteration limit"
    plain_llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    forced_prompt = messages + [HumanMessage(content=(
        f"Stopping further search ({reason}). Based only on what was "
        "actually found above, give your best final answer now. If the "
        "information found does not answer the question, say so honestly "
        "instead of guessing."
    ))]
    final_response = plain_llm.invoke(forced_prompt)
    llm_calls += 1

    print(f"\n({reason} — forced a final answer. {llm_calls} LLM call(s) total, {searches_run} search(es) run)")
    return finalize(final_response.content, sources_seen, chunks_seen)


if __name__ == "__main__":
    print("Agentic RAG (iterative). Type 'exit' to quit.\n")
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
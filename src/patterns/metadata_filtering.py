from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from src.core.retrieved_chunks import get_vectorstore, TOP_K
from src.core.basic_rag import PROMPT, CHAT_MODEL, print_retrieved, build_context

load_dotenv()

KNOWN_FILE_TYPES = {"pdf", "docx", "txt", "md"}


def month_bounds(months_ago: int = 0):
    now = datetime.now()
    year, month = now.year, now.month
    for _ in range(months_ago):
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    start = datetime(year, month, 1)
    end_month, end_year = (month % 12) + 1, year + (1 if month == 12 else 0)
    end = datetime(end_year, end_month, 1)
    return int(start.timestamp()), int(end.timestamp())


def build_where(simple_filters: dict, date_range: tuple = None):
    conditions = [{k: v} for k, v in simple_filters.items() if v not in (None, "")]
    if date_range:
        start_ts, end_ts = date_range
        conditions.append({"first_ingested_at": {"$gte": start_ts}})
        conditions.append({"first_ingested_at": {"$lt": end_ts}})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def parse_filters(raw: str):
    simple_filters, date_range = {}, None
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue

        if "=" in pair:
            key, value = (p.strip() for p in pair.split("=", 1))
        elif pair.lower() in KNOWN_FILE_TYPES:
            key, value = "file_type", pair.lower()
        else:
            key, value = "filename", pair

        if key == "date":
            if value == "this_month":
                date_range = month_bounds(0)
            elif value == "last_month":
                date_range = month_bounds(1)
            else:  # specific day, e.g. 2026-08-15
                day = datetime.strptime(value, "%Y-%m-%d")
                date_range = (int(day.timestamp()), int((day + timedelta(days=1)).timestamp()))
            continue

        simple_filters[key] = int(value) if value.isdigit() else value
    return simple_filters, date_range


def retrieve_with_filters(query: str, simple_filters: dict = None, date_range: tuple = None, top_k: int = TOP_K):
    vectorstore = get_vectorstore()
    where = build_where(simple_filters or {}, date_range)
    return vectorstore.similarity_search_with_score(query, k=top_k, filter=where)


def ask(query: str, simple_filters: dict = None, date_range: tuple = None):
    results = retrieve_with_filters(query, simple_filters, date_range)
    if not results:
        return "No documents found — check ingest.py has run, or your filter matched nothing.", []

    print_retrieved(query, results)
    context = build_context(results)

    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    chain = PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    sources = sorted(set(doc.metadata.get("source", "unknown") for doc, _ in results))
    return answer, sources


if __name__ == "__main__":
    print("RAG query tool with metadata filtering. Type 'exit' to quit.")
    print("Filters: filename, file_type (or just 'pdf'/'txt'/'docx'/'md'), page, date=this_month, date=last_month, date=YYYY-MM-DD\n")
    while True:
        q = input("Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue
        raw_filters = input("Filters (comma-separated — Enter to skip): ").strip()
        simple_filters, date_range = parse_filters(raw_filters) if raw_filters else ({}, None)
        answer, sources = ask(q, simple_filters, date_range)
        print(f"\nAnswer: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()
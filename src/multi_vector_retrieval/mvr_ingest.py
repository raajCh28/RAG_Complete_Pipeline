import os
import glob
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from src.core.prompts import load_prompt

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
COLLECTION_NAME = "multi_vector_collection"

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": lambda p: TextLoader(p, encoding="utf-8"),
    ".docx": Docx2txtLoader,
}

llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

REPRESENTATION_PROMPT = load_prompt("mvr_representation")

representation_chain = REPRESENTATION_PROMPT | llm | StrOutputParser()


def load_documents(data_dir):
    docs = []

    for path in glob.glob(
        os.path.join(data_dir, "**/*"),
        recursive=True
    ):
        if os.path.isdir(path):
            continue

        ext = Path(path).suffix.lower()
        loader_factory = LOADERS.get(ext)

        if loader_factory is None:
            continue

        docs.extend(loader_factory(path).load())

    return docs


def create_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


def generate_representations(chunk):
    response = representation_chain.invoke({
        "chunk": chunk.page_content
    })

    summary = ""
    questions = []
    section = None

    for line in response.splitlines():
        line = line.strip()

        if line == "SUMMARY:":
            section = "summary"

        elif line == "QUESTIONS:":
            section = "questions"

        elif line:
            if section == "summary":
                summary += " " + line

            elif section == "questions":
                question = line.lstrip("1234567890.- ").strip()

                if question:
                    questions.append(question)

    return summary.strip(), questions[:3]


def main():
    print("\nStarting Multi-Vector Ingestion...\n")

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / DATA_DIR

    docs = load_documents(str(data_path))

    if not docs:
        print(f"No documents found in {data_path}.")
        return

    file_count = len(set(
        d.metadata.get("source", "unknown")
        for d in docs
    ))

    print(
        f"Loaded {file_count} file(s) as "
        f"{len(docs)} document/page object(s)."
    )

    chunks = create_chunks(docs)
    print(f"Split into {len(chunks)} chunk(s).")

    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT
    )

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    total_vectors = 0

    for index, chunk in enumerate(chunks, 1):
        print(f"Processing chunk {index}/{len(chunks)}")

        chunk_id = str(uuid.uuid4())

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        page = chunk.metadata.get("page")
        if page is None:
            page = -1

        summary, questions = generate_representations(chunk)

        representations = [
            ("original", chunk.page_content)
        ]

        if summary:
            representations.append(
                ("summary", summary)
            )

        representations.extend(
            ("question", question)
            for question in questions
        )

        texts = [
            text
            for _, text in representations
        ]

        vectors = embeddings.embed_documents(texts)

        ids = [
            f"{chunk_id}_{i}"
            for i in range(len(texts))
        ]

        metadatas = [
            {
                "chunk_id": chunk_id,
                "source": source,
                "page": page,
                "representation": representation,
            }
            for representation, _ in representations
        ]

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        total_vectors += len(texts)

    print("\n" + "=" * 50)
    print("Multi-vector ingestion completed.")
    print(f"Files         : {file_count}")
    print(f"Chunks        : {len(chunks)}")
    print(f"Total vectors : {total_vectors}")
    print(f"Collection    : {COLLECTION_NAME}")
    print("=" * 50)


if __name__ == "__main__":
    main()
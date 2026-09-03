import os
import glob
from pathlib import Path

import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)


# ---------------------------------------------------------
# Normal document loaders
# ---------------------------------------------------------

LOADERS = {
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".docx": lambda path: Docx2txtLoader(path),
}


# ---------------------------------------------------------
# Helper: normalize table cell values
# ---------------------------------------------------------

def clean_cell(value):
    if value is None:
        return ""

    return " ".join(str(value).split())


# ---------------------------------------------------------
# Convert one table row into self-contained text
# ---------------------------------------------------------

def row_to_text(headers, row):
    parts = []

    for header, value in zip(headers, row):
        header = clean_cell(header)
        value = clean_cell(value)

        if not header:
            continue

        if not value:
            continue

        parts.append(f"{header}: {value}")

    return "\n".join(parts)


# ---------------------------------------------------------
# Extract tables from a PDF page
# ---------------------------------------------------------

def extract_tables_from_page(page, source, page_number):
    """
    Detect tables dynamically and convert every table row
    into a separate LangChain Document.

    No column names are hardcoded.
    """

    table_documents = []

    try:
        table_finder = page.find_tables()

    except Exception as exc:
        print(
            f"Warning: Could not detect tables on page "
            f"{page_number} of '{source}': {exc}"
        )
        return table_documents

    tables = table_finder.tables

    for table_index, table in enumerate(tables, start=1):

        try:
            extracted = table.extract()
        except Exception as exc:
            print(
                f"Warning: Could not extract table {table_index} "
                f"from page {page_number} of '{source}': {exc}"
            )
            continue

        if not extracted:
            continue

        # First row is treated as the header.
        headers = extracted[0]

        if not headers:
            continue

        headers = [
            clean_cell(header)
            for header in headers
        ]

        # Skip completely empty headers
        if not any(headers):
            continue

        # -------------------------------------------------
        # Every row becomes one Document
        # -------------------------------------------------

        for row_index, row in enumerate(
            extracted[1:],
            start=1
        ):

            if not row:
                continue

            # Make sure row has the same number of columns
            # as the detected headers.
            row = list(row)

            if len(row) < len(headers):
                row.extend(
                    [""] * (len(headers) - len(row))
                )

            elif len(row) > len(headers):
                row = row[:len(headers)]

            row_text = row_to_text(headers, row)

            if not row_text.strip():
                continue

            document = Document(
                page_content=row_text,
                metadata={
                    "source": source,
                    "page": page_number,
                    "content_type": "table",
                    "table_index": table_index,
                    "row_index": row_index,
                },
            )

            table_documents.append(document)

    return table_documents


# ---------------------------------------------------------
# Extract normal text from a PDF page
# ---------------------------------------------------------

def extract_normal_text_from_page(page, table_bboxes):
    """
    Extract text that is outside detected table areas.

    This prevents table text from being unnecessarily duplicated
    in the normal text chunks.
    """

    blocks = page.get_text("blocks")

    normal_text_parts = []

    for block in blocks:

        if len(block) < 5:
            continue

        x0, y0, x1, y1, text = block[:5]

        if not text or not text.strip():
            continue

        block_rect = fitz.Rect(x0, y0, x1, y1)

        inside_table = False

        for table_bbox in table_bboxes:

            table_rect = fitz.Rect(table_bbox)

            intersection = block_rect & table_rect

            if intersection.is_empty:
                continue

            block_area = block_rect.get_area()

            if block_area <= 0:
                continue

            overlap_ratio = (
                intersection.get_area() / block_area
            )

            # If most of the text block is inside the table,
            # don't include it in normal text.
            if overlap_ratio >= 0.5:
                inside_table = True
                break

        if not inside_table:
            normal_text_parts.append(text.strip())

    return "\n".join(normal_text_parts)


# ---------------------------------------------------------
# PDF loader
# ---------------------------------------------------------

def load_pdf_document(path):
    """
    Load a PDF while preserving BOTH:

    1. Normal text
    2. Table rows

    A PDF may contain:
        text -> table -> text

    on the same page, and all content is preserved.
    """

    documents = []

    try:
        pdf = fitz.open(path)

    except Exception as exc:
        print(f"Error opening PDF '{path}': {exc}")

        # Fallback to normal LangChain PDF loader
        try:
            return PyPDFLoader(path).load()
        except Exception as fallback_exc:
            print(
                f"Fallback PDF loader also failed for "
                f"'{path}': {fallback_exc}"
            )
            return []

    for page_index, page in enumerate(pdf):

        page_number = page_index + 1

        # -------------------------------------------------
        # Detect tables
        # -------------------------------------------------

        table_bboxes = []

        try:
            table_finder = page.find_tables()
            tables = table_finder.tables

            for table in tables:
                table_bboxes.append(table.bbox)

        except Exception as exc:
            print(
                f"Warning: Table detection failed on "
                f"page {page_number} of '{path}': {exc}"
            )
            tables = []

        # -------------------------------------------------
        # Normal text
        # -------------------------------------------------

        normal_text = extract_normal_text_from_page(
            page,
            table_bboxes
        )

        if normal_text.strip():

            documents.append(
                Document(
                    page_content=normal_text,
                    metadata={
                        "source": path,
                        "page": page_number,
                        "content_type": "text",
                    },
                )
            )

        # -------------------------------------------------
        # Tables
        # -------------------------------------------------

        if tables:

            table_documents = extract_tables_from_page(
                page,
                path,
                page_number
            )

            documents.extend(table_documents)

    pdf.close()

    return documents


# ---------------------------------------------------------
# Load one document
# ---------------------------------------------------------

def load_single_document(path):
    """
    Select the appropriate loader based on file extension.
    """

    extension = Path(path).suffix.lower()

    # PDF
    if extension == ".pdf":
        return load_pdf_document(path)

    # TXT / MD / DOCX
    loader_factory = LOADERS.get(extension)

    if loader_factory is None:
        return []

    try:
        loader = loader_factory(path)
        documents = loader.load()

    except Exception as exc:
        print(f"Error loading '{path}': {exc}")
        return []

    # Add content_type metadata to normal documents
    for document in documents:

        document.metadata["content_type"] = "text"

    return documents


# ---------------------------------------------------------
# Load all documents recursively
# ---------------------------------------------------------

def load_documents(data_dir: str):
    """
    Recursively load all supported documents from data_dir.
    """

    documents = []

    for path in glob.glob(
        os.path.join(data_dir, "**/*"),
        recursive=True
    ):

        if os.path.isdir(path):
            continue

        extension = Path(path).suffix.lower()

        supported_extensions = {
            ".pdf",
            ".txt",
            ".md",
            ".docx",
        }

        if extension not in supported_extensions:
            continue

        loaded_documents = load_single_document(path)

        documents.extend(loaded_documents)

    return documents
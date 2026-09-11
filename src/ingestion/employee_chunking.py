import re

from langchain_core.documents import Document


EMPLOYEE_RECORD_PATTERN = re.compile(
    r"(?ms)^Employee\s+(?P<number>\d+)\s*\n"
    r"(?P<record>.*?)(?=^Employee\s+\d+\s*$|\Z)"
)
EMPLOYEE_ID_PATTERN = re.compile(
    r"(?mi)^Employee ID:\s*(?P<employee_id>EMP\d+)\s*$"
)


def is_employee_directory_page(document: Document) -> bool:
    """Return True only for pages containing structured employee records."""
    return (
        document.metadata.get("category") == "employee"
        and document.metadata.get("filename") == "employees.pdf"
    )


def split_employee_records(documents: list[Document]) -> list[Document]:
    """Split employee-directory pages into one focused document per employee.

    Documents outside the employee directory are returned unchanged so the
    normal ingestion splitter continues to handle policies and other files.
    """
    split_documents = []

    for document in documents:
        if not is_employee_directory_page(document):
            split_documents.append(document)
            continue

        matches = list(EMPLOYEE_RECORD_PATTERN.finditer(document.page_content))
        if not matches:
            split_documents.append(document)
            continue

        for match in matches:
            record = match.group("record").strip()
            employee_match = EMPLOYEE_ID_PATTERN.search(record)

            if not employee_match:
                continue

            employee_id = employee_match.group("employee_id").upper()
            split_documents.append(
                Document(
                    page_content=(
                        f"Employee {match.group('number')}\n"
                        f"{record}"
                    ),
                    metadata={
                        **document.metadata,
                        "employee_id": employee_id,
                        "entity_type": "employee",
                    },
                )
            )

    return split_documents
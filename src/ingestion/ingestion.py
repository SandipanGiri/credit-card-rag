import os
import pathlib
from dotenv import load_dotenv
from sqlalchemy import text
from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document
from src.core.rdbm import get_vector_store
from src.ingestion.ingestion_rerank import ingest_file
from pathlib import Path

load_dotenv()

_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split a long string into overlapping character windows."""
    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step
    return chunks


def file_check(file_path):
    print("am inside file check verification")
    print("incoming file path:", file_path)

    # Keep the same relative path format used during ingestion
    source_path = str(Path(str(file_path))).replace("\\", "/")

    print("Source path for checking:", source_path)

    vector_store = get_vector_store(collection_name="multimodal_chunks")

    with vector_store._engine.connect() as conn:

        result = conn.execute(
            text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM langchain_pg_embedding
                    WHERE REPLACE(
                        cmetadata->>'source',
                        CHR(92),
                        '/'
                    ) = :source
                )
            """),
            {"source": source_path},
        )

        existing = result.scalar()

    print("After verification:", existing)
    return bool(existing)


def run_ingestion(file_path: str) -> dict:
    """Run the full ingestion pipeline for a single word file.
    Args:
        file_path: Absolute or relative path to the source word file.
    Returns:
        Dict with "status", "doc_id", and "chunks_ingested" count.
    """
    print("Ingestion Started")
    existing = file_check(file_path)

    if existing:
        message = f"{Path(file_path).name} already exists. Skipping ingestion."
        print("*****", message)
        return {
            "status": "unsuccessful",
            "message": "File already exists",
        }

    else:
        # 1 multimodla ingetsion
        resolved = pathlib.Path(file_path).resolve()

        doc_id = upsert_document(resolved.name, str(resolved))
        print(f"[ingestion] doc_id={doc_id}  file={file_path}")

        print(f"[ingestion] Parsing: {file_path}")
        parsed_elements = parse_document(file_path)
        print(f"[ingestion] Docling produced {len(parsed_elements)} elements")

        chunks: list[dict] = []
        for elem in parsed_elements:
            if (
                elem["content_type"] == "text"
                and len(elem["content"]) > _TEXT_CHUNK_SIZE
            ):
                for sub in _split_text(
                    elem["content"], _TEXT_CHUNK_SIZE, _TEXT_CHUNK_OVERLAP
                ):
                    # Each sub-chunk inherits the parent element's full metadata
                    chunks.append(
                        {
                            "content": sub,
                            "content_type": elem["content_type"],
                            "metadata": elem["metadata"],
                        }
                    )
            else:
                chunks.append(elem)

        print(f"[ingestion] {len(chunks)} chunks ready for embedding")

        count = store_chunks(chunks, doc_id)
        print(f"[ingestion] Stored {count} chunks → multimodal_chunks")

        # # 2. Rerank ingestion
        # # --------------------------------------------------
        # print("[ingestion] Starting rerank ingestion...")

        # ingest_file(file_path)

        # print("[ingestion] Rerank ingestion completed successfully")

        # --------------------------------------------------
        # 3. Everything succeeded
        # --------------------------------------------------

        return {
            "status": "success",
            "message": "File ingested successfully",
            "doc_id": doc_id,
            "chunks_ingested": count,
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2:
        file_path = pathlib.Path(sys.argv[1])
    else:
        file_path = pathlib.Path("data/KB_Credit_Card_Spend_Summarizer.docx")

    if not file_path.exists():
        raise FileNotFoundError(f"word file was not found at: {file_path.resolve()}")

    result = run_ingestion(str(file_path))
    print(f"\nIngestion complete: {result}")

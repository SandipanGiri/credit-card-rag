import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredWordDocumentLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.core.rdbm import get_vector_store
from sqlalchemy import create_engine, text

load_dotenv()
PG_CONNECTION = os.getenv("PG_CONNECTION_STRING")


def load_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == ".docx" or ext == ".doc":
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    return loader.load(), ext


def index_add():
    engine = create_engine(os.getenv("PG_CONNECTION_STRING"))
    with engine.connect() as conn:
        conn.execute(
            text(
                "ALTER TABLE langchain_pg_embedding ALTER COLUMN embedding TYPE vector(1536)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
        )
        conn.commit()


def ingest_file(file_path):
    docs, ext = load_document(file_path)
    print("docs", docs)
    print("Pages: " + str(len(docs)))

    for doc in docs:
        doc.metadata.update(
            {
                "source": file_path,
                "document_name": os.path.basename(file_path),
                "document_extension": ext,
                "page": doc.metadata.get("page", None),
                "category": "Credit_card_summariser",
                "last_updated": os.path.getmtime(file_path),
            }
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(docs)
    print("Chunks: " + str(len(chunks)))

    vector_store = get_vector_store("RerankingRAGVectorStore")
    vector_store.add_documents(chunks)
    index_add()
    print("==== Ingestion completed ====")


if __name__ == "__main__":
    # ingest_pdf("data/HR_Knowledge_Base_2025.pdf")
    ingest_file("data/KB_Credit_Card_Spend_Summarizer.docx")

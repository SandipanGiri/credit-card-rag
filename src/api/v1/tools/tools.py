from src.api.v1.states.rag_state import RAGState
from src.core.db import get_db_conn, _embed_texts, similarity_search
import re
import psycopg
import os
from psycopg.rows import dict_row

# from langchain_core.tools import tool
from sqlalchemy import text
from langchain_core.documents import Document
import pathlib
import base64
from src.core.rdbm import get_vector_store

_raw_conn = os.getenv("PG_CONNECTION_STRING_FTS")


# def vector_search_node(state: RAGState):
#     """
#     this function is used to find the similar text using the similarity_Search method
#     """
#     print("====== INSIDE vector_search_node: searching the vector db")
#     vector_store = get_vector_store()
#     docs = vector_store.similarity_search(state["query"], k=20)
#     print(
#         "======= INSIDE vector_search_node: Searched the Vector DB - Retrieved Docs Count:",
#         len(docs),
#     )
#     return {**state, "retrieved_docs": docs}


def vector_search_node(state: RAGState):
    print("====== INSIDE vector_search_node: searching the vector db")

    results = similarity_search(
        query=state["query"],
        k=20,
    )

    print(
        "======= VECTOR SEARCH: Retrieved Docs Count:",
        len(results),
    )

    docs = []

    for result in results:
        doc = Document(
            page_content=result.get("content") or "",
            metadata={
                "chunk_type": result.get("chunk_type"),
                "page_number": result.get("page_number"),
                "section": result.get("section"),
                "source_file": result.get("source_file"),
                "element_type": result.get("element_type"),
                "mime_type": result.get("mime_type"),
                "position": result.get("position"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity"),
            },
        )

        docs.append(doc)

    return {
        **state,
        "retrieved_docs": docs,
    }


# def fts_search(query: str, k: int = 20):
#     print("====== FTS SEARCH ======")
#     sql = """
#             SELECT
#                 id,
#                 content,
#                 metadata,
#                 ts_rank_cd(
#                     to_tsvector('english', content),
#                     plainto_tsquery('english', %s)
#                 ) AS score
#             FROM multimodal_chunks
#             WHERE to_tsvector('english', content)
#                 @@ plainto_tsquery('english', %s)
#             ORDER BY score DESC
#             LIMIT %s
#     """
#     docs = []
#     with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
#         with conn.cursor() as cur:
#             cur.execute(sql, (query, query, k))
#             rows = cur.fetchall()
#     for row in rows:
#         docs.append(
#             Document(page_content=row["content"], metadata=row["metadata"] or {})
#         )

#     return docs


def fts_search(query: str, k: int = 20):
    """Perform PostgreSQL full-text search against multimodal_chunks."""
    print("====== FTS SEARCH ======")

    sql = """
        SELECT
            id,
            doc_id,
            content,
            chunk_type,
            element_type,
            page_number,
            section,
            source_file,
            metadata,
            ts_rank_cd(
                to_tsvector('english', COALESCE(content, '')),
                plainto_tsquery('english', %s)
            ) AS score
        FROM multimodal_chunks
        WHERE to_tsvector(
            'english',
            COALESCE(content, '')
        ) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
    """

    docs = []

    with psycopg.connect(
        _raw_conn,
        row_factory=dict_row,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (query, query, k),
            )

            rows = cur.fetchall()

    for row in rows:
        metadata = row["metadata"] or {}

        # Add database identifiers and useful chunk information
        # to the LangChain Document metadata.
        metadata = {
            **metadata,
            "id": row["id"],
            "doc_id": str(row["doc_id"]) if row["doc_id"] else None,
            "chunk_type": row["chunk_type"],
            "element_type": row["element_type"],
            "page_number": row["page_number"],
            "section": row["section"],
            "source_file": row["source_file"],
            "fts_score": float(row["score"] or 0),
        }

        docs.append(
            Document(
                page_content=row["content"] or "",
                metadata=metadata,
            )
        )

    print("FTS results:", len(docs))

    return docs


def fts_search_node(state: RAGState):
    """this does full text search"""
    print("====== INSIDE fts_search_node")
    docs = fts_search(state["query"], k=20)
    print("======= FTS Search:", len(docs))
    return {**state, "retrieved_docs": docs}
    # return process_multimodal_docs(state, docs)


# fixing the issue with vector search non retreival
# def get_document_id(doc):
#     return (
#         doc.metadata.get("uuid") or doc.metadata.get("source") or hash(doc.page_content)
#     )
def get_document_id(doc):
    return (
        doc.metadata.get("doc_id")
        or doc.metadata.get("id")
        or doc.metadata.get("uuid")
        or doc.metadata.get("source")
        or hash(doc.page_content)
    )


# def hybrid_search_node(state: RAGState):
#     """this does hybrid search"""
#     print("====== INSIDE hybrid_search_node")
#     vector_store = get_vector_store()
#     vector_docs = vector_store.similarity_search(state["query"], k=20)
#     fts_docs = fts_search(state["query"], k=20)
#     scores = {}
#     docs = {}
#     RRF_K = 60
#     for rank, doc in enumerate(vector_docs, 1):
#         doc_id = get_document_id(doc)
#         scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
#         docs[doc_id] = doc
#     for rank, doc in enumerate(fts_docs, 1):
#         doc_id = get_document_id(doc)
#         scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
#         docs[doc_id] = doc

#     ranked = sorted(scores, key=scores.get, reverse=True)
#     final_docs = []
#     for doc_id in ranked[:20]:
#         doc = docs[doc_id]
#         doc.metadata["rrf_score"] = scores[doc_id]
#         final_docs.append(doc)

#     print("Hybrid results:", len(final_docs))
#     return {**state, "retrieved_docs": final_docs}
# return process_multimodal_docs(state, final_docs)


def hybrid_search_node(state: RAGState):
    """Perform hybrid vector + full-text search using Reciprocal Rank Fusion."""
    print("====== INSIDE hybrid_search_node")

    # ---------------------------------------------------------
    # 1. Vector search
    # Uses the custom similarity_search() that queries
    # multimodal_chunks directly.
    # ---------------------------------------------------------
    vector_results = similarity_search(
        query=state["query"],
        k=20,
    )

    vector_docs = []

    for result in vector_results:
        doc = Document(
            page_content=result.get("content") or "",
            metadata={
                "chunk_type": result.get("chunk_type"),
                "page_number": result.get("page_number"),
                "section": result.get("section"),
                "source_file": result.get("source_file"),
                "element_type": result.get("element_type"),
                "mime_type": result.get("mime_type"),
                "position": result.get("position"),
                "metadata": result.get("metadata") or {},
                "similarity": result.get("similarity"),
            },
        )

        vector_docs.append(doc)

    print("Vector results:", len(vector_docs))

    # ---------------------------------------------------------
    # 2. Full-text search
    # ---------------------------------------------------------
    fts_docs = fts_search(
        state["query"],
        k=20,
    )

    print("FTS results:", len(fts_docs))

    # ---------------------------------------------------------
    # 3. Reciprocal Rank Fusion (RRF)
    # ---------------------------------------------------------
    scores = {}
    docs = {}

    RRF_K = 60

    # Vector search rankings
    for rank, doc in enumerate(vector_docs, 1):
        doc_id = get_document_id(doc)

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        docs[doc_id] = doc

    # FTS rankings
    for rank, doc in enumerate(fts_docs, 1):
        doc_id = get_document_id(doc)

        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)

        # Keep the document already stored, or use FTS document
        # if this is a new document.
        docs[doc_id] = docs.get(doc_id, doc)

    # ---------------------------------------------------------
    # 4. Sort by RRF score
    # ---------------------------------------------------------
    ranked = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    # ---------------------------------------------------------
    # 5. Return top 20 documents
    # ---------------------------------------------------------
    final_docs = []

    for doc_id in ranked[:20]:
        doc = docs[doc_id]

        # Don't mutate the original metadata unexpectedly
        doc.metadata = {
            **doc.metadata,
            "rrf_score": scores[doc_id],
        }

        final_docs.append(doc)

    print("Hybrid results:", len(final_docs))

    return {
        **state,
        "retrieved_docs": final_docs,
    }


# def process_multimodal_docs(state, docs):

#     text_docs = []
#     table_docs = []
#     image_docs = []

#     for doc in docs:

#         doc_type = doc.metadata.get("type", "text")

#         if doc_type == "image":
#             image_docs.append(doc)

#         elif doc_type == "table":
#             table_docs.append(doc)

#         else:
#             text_docs.append(doc)

#     return {
#         **state,
#         "retrieved_docs": docs,
#         "text_docs": text_docs,
#         "table_docs": table_docs,
#         "image_docs": image_docs,
#     }

#        chunk_type="image"


# def extract_images_node(state: RAGState):

#     print("========== INSIDE IMAGE SEARCH NODE ==========")

#     query = state["query"].lower()

#     if any(word in query for word in ["all", "available", "list"]):
#         k = 20

#     else:
#         # retrieve more candidates
#         k = 5

#     results = similarity_search(query=state["query"], k=k, chunk_type="image")

#     images = []

#     for result in results:

#         if result.get("image_base64"):

#             images.append(
#                 {
#                     "content": result["image_base64"],
#                     "mime_type": result.get("mime_type"),
#                     "source_file": result.get("source_file"),
#                     "page_number": result.get("page_number"),
#                 }
#             )

#     return {**state, "images": images}


def extract_images_node(state: RAGState):
    """Retrieve image chunks relevant to the user's query."""

    print("========== INSIDE IMAGE SEARCH NODE ==========")

    query = state["query"]

    # Retrieve more candidates when the user asks for all/available/list.
    if any(word in query.lower() for word in ["all", "available", "list"]):
        k = 20
    else:
        k = 5

    print("Image search query:", query)
    print("Image search k:", k)

    results = similarity_search(
        query=query,
        k=k,
        chunk_type="image",
    )

    print("Image chunks retrieved:", len(results))

    images = []

    for result in results:
        image_base64 = result.get("image_base64")

        if not image_base64:
            continue

        images.append(
            {
                "content": image_base64,
                "mime_type": result.get("mime_type"),
                "source_file": result.get("source_file"),
                "page_number": result.get("page_number"),
                "section": result.get("section"),
                "chunk_type": result.get("chunk_type"),
                "element_type": result.get("element_type"),
                "similarity": result.get("similarity"),
                "metadata": result.get("metadata") or {},
            }
        )

    print("Images with valid base64 data:", len(images))

    return {
        **state,
        "images": images,
    }

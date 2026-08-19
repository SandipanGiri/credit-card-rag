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


def vector_search_node(state: RAGState):
    """
    this function is used to find the similar text using the similarity_Search method
    """
    print("====== INSIDE vector_search_node: searching the vector db")
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(state["query"], k=20)
    print(
        "======= INSIDE vector_search_node: Searched the Vector DB - Retrieved Docs Count:",
        len(docs),
    )
    return {**state, "retrieved_docs": docs}


def fts_search(query: str, k: int = 20):
    print("====== FTS SEARCH ======")
    sql = """
            SELECT
                id,
                content,
                metadata,
                ts_rank_cd(
                    to_tsvector('english', content),
                    plainto_tsquery('english', %s)
                ) AS score
            FROM multimodal_chunks
            WHERE to_tsvector('english', content)
                @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
    """
    docs = []
    with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query, query, k))
            rows = cur.fetchall()
    for row in rows:
        docs.append(
            Document(page_content=row["content"], metadata=row["metadata"] or {})
        )

    return docs


def fts_search_node(state: RAGState):
    """this does full text search"""
    print("====== INSIDE fts_search_node")
    docs = fts_search(state["query"], k=20)
    print("======= FTS Search:", len(docs))
    return {**state, "retrieved_docs": docs}
    # return process_multimodal_docs(state, docs)


def get_document_id(doc):
    return (
        doc.metadata.get("uuid") or doc.metadata.get("source") or hash(doc.page_content)
    )


def hybrid_search_node(state: RAGState):
    """this does hybrid search"""
    print("====== INSIDE hybrid_search_node")
    vector_store = get_vector_store()
    vector_docs = vector_store.similarity_search(state["query"], k=20)
    fts_docs = fts_search(state["query"], k=20)
    scores = {}
    docs = {}
    RRF_K = 60
    for rank, doc in enumerate(vector_docs, 1):
        doc_id = get_document_id(doc)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        docs[doc_id] = doc
    for rank, doc in enumerate(fts_docs, 1):
        doc_id = get_document_id(doc)
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (RRF_K + rank)
        docs[doc_id] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    final_docs = []
    for doc_id in ranked[:20]:
        doc = docs[doc_id]
        doc.metadata["rrf_score"] = scores[doc_id]
        final_docs.append(doc)

    print("Hybrid results:", len(final_docs))
    return {**state, "retrieved_docs": final_docs}
    # return process_multimodal_docs(state, final_docs)


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


def extract_images_node(state: RAGState):

    print("========== INSIDE IMAGE SEARCH NODE ==========")

    query = state["query"].lower()

    if any(word in query for word in ["all", "available", "list"]):
        k = 20

    else:
        # retrieve more candidates
        k = 5

    results = similarity_search(query=state["query"], k=k, chunk_type="image")

    images = []

    for result in results:

        if result.get("image_base64"):

            images.append(
                {
                    "content": result["image_base64"],
                    "mime_type": result.get("mime_type"),
                    "source_file": result.get("source_file"),
                    "page_number": result.get("page_number"),
                }
            )

    return {**state, "images": images}

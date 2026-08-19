from fastapi import APIRouter, HTTPException
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse
from src.api.v1.services.query_service import query_documents, query_documents_stream
from fastapi.responses import StreamingResponse
from src.core.guardrails import GuardrailViolation
import json
import base64

router = APIRouter(prefix="/api/v1/query")


def encode_image(image_path: str):

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")

    return encoded


# for non streaming repsonse
@router.post("/")
def query_endpoint(request: QueryRequest) -> QueryResponse:
    try:
        response = query_documents(request.query, request.thread_id)
        print("SERVICE RESPONSE:", response)
        # return response
    except GuardrailViolation as violation:
        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )
    # return docs
    # print("REQUEST THREAD ID:", request.thread_id)
    # print("GRAPH RESPONSE:", response)
    if response is None:
        raise HTTPException(status_code=500, detail="Agent returned empty response")

    images = []

    for image_item in response.get("images", []):

        try:

            if isinstance(image_item, dict):
                image_path = (
                    image_item.get("image_path")
                    or image_item.get("path")
                    or image_item.get("file_path")
                )

            else:
                image_path = image_item

            if image_path:
                images.append(encode_image(image_path))

        except Exception as e:
            print("Image encoding failed:", e)

    return QueryResponse(
        query=request.query,
        thread_id=request.thread_id,
        answer=response.get("answer", ""),
        policy_citations=response.get("policy_citations", ""),
        page_no=response.get("page_no", ""),
        document_name=response.get("document_name", ""),
        sql_query_executed=response.get("sql_query_executed"),
        images=response.get("images", []),
    )

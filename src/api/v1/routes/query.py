from fastapi import APIRouter, HTTPException, Request
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


# -----------------------------
# Non streaming response
# -----------------------------


@router.post("/")
async def query_endpoint(request: Request, body: QueryRequest) -> QueryResponse:

    try:

        response = await query_documents(request, body.query, body.thread_id)

        print("SERVICE RESPONSE:", response)

    except GuardrailViolation as violation:

        raise HTTPException(
            status_code=400,
            detail={"guardrail": violation.guard, "message": violation.message},
        )

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))

    if response is None:

        raise HTTPException(status_code=500, detail="Agent returned empty response")

    if not isinstance(response, dict):

        raise HTTPException(
            status_code=500, detail="Invalid response format from agent"
        )

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
        query=body.query,
        thread_id=body.thread_id,
        answer=response.get("answer", ""),
        policy_citations=response.get("policy_citations", ""),
        page_no=response.get("page_no", ""),
        document_name=response.get("document_name", ""),
        sql_query_executed=response.get("sql_query_executed"),
        images=images,
    )


# -----------------------------
# Streaming response
# -----------------------------


@router.post("/stream")
async def stream_query_endpoint(request: Request, body: QueryRequest):

    async def event_generator():

        try:

            async for chunk in query_documents_stream(
                request, body.query, body.thread_id
            ):

                # token streaming

                if isinstance(chunk, str):

                    yield (
                        "event: token\n" f"data: {json.dumps({'content': chunk})}\n\n"
                    )

                elif isinstance(chunk, dict):

                    # token

                    if chunk.get("content"):

                        yield (
                            "event: token\n"
                            f"data: {json.dumps({'content': chunk['content']})}\n\n"
                        )

                    # metadata

                    if chunk.get("done"):

                        metadata = {
                            "done": True,
                            "sources": chunk.get("sources", []),
                            "answer": chunk.get("answer", ""),
                            "images": chunk.get("images", []),
                            "policy_citations": chunk.get("policy_citations", ""),
                            "page_no": chunk.get("page_no", ""),
                            "document_name": chunk.get("document_name", ""),
                        }

                        yield ("event: metadata\n" f"data: {json.dumps(metadata)}\n\n")

            yield ("event: done\n" f"data: {json.dumps({'status':'completed'})}\n\n")

        except GuardrailViolation as violation:

            yield ("event: guardrail_error\n" f"data: {json.dumps({
                    'guardrail': violation.guard,
                    'message': violation.message
                })}\n\n")

        except Exception as e:

            yield ("event: error\n" f"data: {json.dumps({
                    'message': str(e)
                })}\n\n")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

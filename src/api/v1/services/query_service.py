from fastapi import Request
from src.core.guardrails import guard_input, guard_output
import json

# -----------------------------
# Non streaming
# -----------------------------


async def query_documents(request: Request, query: str, thread_id: str):

    print(query)

    try:

        # Input guardrail
        guard_input(query)

        # get graph from FastAPI app state
        rag_graph = request.app.state.rag_graph

        initial_state = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "response": {},
            "images": [],
            "evaluation": {},
            "is_good": False,
            "attempts": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        result = await rag_graph.ainvoke(initial_state, config=config)

        response = result.get("response", {})

        # Output guardrail
        if response.get("answer"):

            response["answer"] = guard_output(response["answer"])

        return response

    except Exception as e:

        print(f"Error in query_documents: {e}")

        raise


# -----------------------------
# Streaming
# -----------------------------


# async def query_documents_stream(request: Request, query: str, thread_id: str):

#     try:

#         print(query)

#         # Input guardrail
#         guard_input(query)

#         rag_graph = request.app.state.rag_graph

#         initial_state = {
#             "query": query,
#             "retrieved_docs": [],
#             "reranked_docs": [],
#             "response": {},
#             "images": [],
#             "evaluation": {},
#             "is_good": False,
#             "attempts": 0,
#         }

#         config = {"configurable": {"thread_id": thread_id}}

#         async for event in rag_graph.astream_events(
#             initial_state, config=config, version="v2"
#         ):

#             kind = event.get("event")

#             # LLM token streaming

#             if kind == "on_chat_model_stream":

#                 chunk = event["data"]["chunk"]

#                 content = chunk.content

#                 if content:

#                     # output guardrail
#                     content = guard_output(content)

#                     yield {"content": content}

#         # after stream completed get state

#         final_state = await rag_graph.get_state(config)

#         values = final_state.values

#         yield {
#             "done": True,
#             "sources": values.get("sources", []),
#             "images": values.get("images", []),
#             "policy_citations": values.get("policy_citations", ""),
#             "page_no": values.get("page_no", ""),
#             "document_name": values.get("document_name", ""),
#         }

#     except Exception as e:

#         print(f"Streaming error: {e}")

#         yield {"error": str(e)}


async def query_documents_stream(request: Request, query: str, thread_id: str):

    try:

        print(query)

        # Input guardrail
        guard_input(query)

        rag_graph = request.app.state.rag_graph

        initial_state = {
            "query": query,
            "retrieved_docs": [],
            "reranked_docs": [],
            "response": {},
            "images": [],
            "evaluation": {},
            "is_good": False,
            "attempts": 0,
        }

        config = {"configurable": {"thread_id": thread_id}}

        full_response = ""

        # Stream LLM tokens
        async for event in rag_graph.astream_events(
            initial_state, config=config, version="v2"
        ):

            kind = event.get("event")

            if kind == "on_chat_model_stream":

                node_name = event.get("metadata", {}).get("langgraph_node")

                # only stream final response nodes
                if node_name not in ["chat_response", "generate_answer"]:
                    continue

                chunk = event["data"]["chunk"]

                content = chunk.content

                if content:

                    # collect complete answer
                    full_response += content

                    # stream raw tokens
                    yield {"content": content}

        # ===============================
        # AFTER STREAM COMPLETES
        # Apply output guardrail
        # ===============================

        guarded_response = guard_output(full_response)

        # Get final graph state asynchronously
        final_state = await rag_graph.aget_state(config)

        values = final_state.values
        # final_answer = values.get("answer", "")
        final_answer = values.get("response", {}).get("answer", "")
        # print("========== FINAL STATE ==========")
        # print(values)

        # print("========== ANSWER FROM STATE ==========")
        # print(values.get("answer"))

        # Send final guarded response metadata

        if values.get("intent") == "CHITCHAT":

            yield {"done": True, "answer": guarded_response}

        else:

            yield {
                # "answer": values.get("answer", ""),
                "done": True,
                "answer": final_answer,
                "sources": values.get("sources", []),
                "images": values.get("images", []),
                "policy_citations": values.get("policy_citations", ""),
                "page_no": values.get("page_no", ""),
                "document_name": values.get("document_name", ""),
            }

    except Exception as e:

        print("Streaming error:", e)

        yield {"error": str(e)}

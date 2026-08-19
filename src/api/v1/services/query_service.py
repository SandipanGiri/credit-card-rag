from src.api.v1.agents.agents import run_search_agent_stream, run_search_agent
from src.core.guardrails import guard_input, guard_output


# for non streaming response
def query_documents(query: str, thread_id: str):
    # query=request["query"]
    print(query)
    try:
        # inout guardrails toxicity
        guard_input(query)
        # return run_search_agent(query)
        result = run_search_agent(query, thread_id)
        if isinstance(result, dict) and result.get("answer"):
            # output guard rail for PII
            result["answer"] = guard_output(result["answer"])
            print("results readacted", result["answer"])
        return result
    except Exception as e:
        print(f"Error in query_documents: {e}")
        raise


# method for streaming response
async def query_documents_stream(query: str, thread_id: str):
    try:
        print(query)

        # Input guardrail
        guard_input(query)

        # Stream agent response
        async for chunk in run_search_agent_stream(query, thread_id):

            # If chunk is a dict, get response text
            if isinstance(chunk, dict):
                response = chunk.get("response", "")
            else:
                response = str(chunk)

            # Output guardrail
            if response:
                response = guard_output(response)

            # SSE format
            yield f"data: {response}\n\n"

    except GuardrailViolation as violation:
        yield ("event: guardrail_error\n" f"data: {json.dumps({
                'guardrail': violation.guard,
                'message': violation.message})}\n\n")

    except Exception as e:
        print(f"Streaming error: {e}")

        yield ("event: error\n" f"data: {json.dumps({'message': str(e)})}\n\n")

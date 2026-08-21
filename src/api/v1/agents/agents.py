# nodes we want
# 1. vector_search (top-k=20)
# 2. rerank
# 3. generate_answer
import os
import cohere
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel
from typing import Literal
from src.api.v1.states.rag_state import RAGState
from langgraph.checkpoint.memory import InMemorySaver
from src.api.v1.tools.tools import (
    vector_search_node,
    fts_search_node,
    hybrid_search_node,
    extract_images_node,
)
from src.api.v1.schemas.query_schema import AIResponse, EvaluationResult
from src.core.rdbm import get_sql_database
import traceback
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

load_dotenv()
# CHECKPOINT_DB_URI = os.getenv("LANGGRAPH_CHECKPOINT_DB_URI")
# checkpoint_cm = AsyncPostgresSaver.from_conn_string(CHECKPOINT_DB_URI)
# checkpoint = checkpoint_cm.__aenter__()
# checkpoint.setup()


def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL"), api_key=os.getenv("OPENAI_API_KEY")
    )


class RouteDecision(BaseModel):
    route: Literal["VECTOR_DB", "HYBRID", "FTS", "RDBMS", "IMAGE"]
    reason: str  # for debugging


class IntentDecision(BaseModel):
    intent: Literal["CHITCHAT", "REPHRASER"]
    reason: str


def intent_check_node(state: RAGState):

    print("========== INSIDE intent_check_node ==========")

    llm = _get_llm()

    structured_llm = llm.with_structured_output(IntentDecision)

    # Get conversation history
    conversation_history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an intent classifier.

                Classify the user message into one of two categories.

                CHITCHAT:
                - hi
                - hello
                - hey
                - good morning
                - good evening
                - thanks
                - thank you
                - how are you
                - who are you
                - casual conversation
                

                Conversation behavior rules:

                    1. Greeting handling:
                    - If the user only sends a greeting such as "hi", "hello", "hey", "good morning", or similar:
                    - Respond politely.
                    -Politely reject any questions that is asked of out of credit card informations 
                    - Keep the response brief.
                    - Ask what the user needs help with.

                    Examples:
                    User: "Hi"
                    Assistant: "Hi! How can I help you today?"

                    User: "Good evening"
                    Assistant: "Good evening! How can I assist you?"

                    2. Normal conversation:
                    - After the greeting exchange, answer the user's questions normally.
                    - Do not repeat greeting responses on every message.
                    - Maintain context from previous messages.
                    - Provide accurate, useful, and clear answers.
                    - Ask clarification questions when the user's request is unclear.


                REPHRASER:
                - credit card questions
                - policy questions
                - document questions
                - product questions
                - transaction questions
                - any information request

                Return ONLY one exact value:
                CHITCHAT
                or
                REPHRASER

                Do not shorten, modify, or create new values.
                
                """,
            ),
            (
                "human",
                """
                User message:
                {query}
                """,
            ),
        ]
    )

    result = (prompt | structured_llm).invoke(
        {"history": conversation_history, "query": state["query"]}
    )

    print(f"[intent_check_node] Intent: {result.intent}, Reason: {result.reason}")

    return {"intent": result.intent}


def chat_response_node(state: RAGState):

    print("========== INSIDE chat_response_node ==========")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a friendly credit card assistant.

                Rules:
                - Reply briefly and politely.
                - Maintain conversation context.
                - Do not repeat greetings every time.
                - If the question is outside credit card assistance,
                  politely say you can only help with credit card information.

                Examples:

                User: Hi
                Assistant: Hi! How can I help you today?

                User: How are you?
                Assistant: I'm doing great! How can I assist you?

                User: Thanks
                Assistant: You're welcome!
                """,
            ),
            (
                "human",
                """
                Conversation history:

                {history}


                Latest user message:

                {query}
                """,
            ),
        ]
    )

    # Build previous conversation
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    result = (prompt | llm).invoke({"history": history, "query": state["query"]})

    response = {
        "answer": result.content,
        "policy_citations": "",
        "page_no": "",
        "document_name": "",
    }

    return {
        "answer": result.content,
        # "response": response,
        # append AI message to checkpoint memory
        "messages": [AIMessage(content=result.content)],
    }


def add_user_message_node(state: RAGState):

    return {
        **state,
        "messages": state.get("messages", []) + [HumanMessage(content=state["query"])],
    }


def rephraser_node(state: RAGState) -> RAGState:
    """
    Rephrase the current user query using previous conversation
    so that it becomes a standalone query suitable for retrieval.
    """

    print("========== INSIDE rephraser_node ==========")

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a query rephraser for a RAG system.

                Your job is to rephrase the current user question
                into a standalone question using the previous conversation.

                Rules:
                1. Preserve the user's original intent.
                2. Resolve references such as:
                   - it
                   - this
                   - that
                   - they
                   - them
                   - above
                   - previous
                   - same
                3. Do not answer the question.
                4. Do not add information that is not present
                   in the conversation.
                5. If the question is already standalone,
                   return it unchanged.
                6. Return ONLY the rephrased question.
                """,
            ),
            (
                "human",
                """
                Previous conversation:
                {history}

                Current user question:
                {query}
                """,
            ),
        ]
    )

    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )

    chain = prompt | llm

    result = chain.invoke({"history": history, "query": state["query"]})

    rephrased_query = result.content.strip()

    print(f"[Rephraser] Original: {state['query']}")

    print(f"[Rephraser] Rephrased: {rephrased_query}")

    return {**state, "query": rephrased_query}


def router_node(state: RAGState) -> RAGState:
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a helpful credit_card assistant for an Agentic RAG System.

                   -Answer the user's question using only the
                       provided context 
                    -politely reject if any question asked out of scope
                    -don't answer out of this .

                    -Classify the user's query into EXACTLY one of the following routes: 

                     'VECTOR_DB' -  the auery asks about policies, procedures, guides, guidelines,
                      regulations, or any topic that requires reading text documents 

                       'HYBRID' - Use when BOTH semantic understanding AND keyword matching are
                            important.

                            Use HYBRID when:
                            - the query contains important keywords AND
                            - the query also requires understanding the meaning/context of those
                            keywords.

                       'FTS' -
                        Use when the query contains specific keywords, names, phrases,
                        document terms, identifiers, or exact text that should be matched
                        lexically. 

                      'RDBMS' - the query asks about products, product prices, stock/inventory,
                      product categories, customer orders, order items, or anything answerable
                      from a structrured e-commerce database tables:
                      products, categories, orders, order_items

                      'IMAGE':
                        - User requests images, pictures, photos, diagrams, visual assets

                     Reply with the route and one sentence of reason.
                   """,
            ),
            (
                "human",
                """
                   Question:
                   {query}
                """,
            ),
        ]
    )

    chain = prompt | structured_llm
    decision = chain.invoke({"query": state["query"]})
    print(f"[router_node's decision]: {decision.route} and reason: {decision.reason}")

    return {**state, "route": decision.route}


def nl2sql_node(state: RAGState) -> RAGState:
    print("About to generate nl2sql")
    # connect to LLM
    llm = _get_llm()
    # connect to rdbms
    db = get_sql_database()
    # get the tables' live schema
    schema_info = db.get_table_info()
    # write the system prompt and pass on the schema to get only sql query
    sql_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a PostgreSQL expert. Given the database schema below,
                   write a single valid SELECT query that answers the user's question.


                   Rules:
                   - Return ONLY the raw SQL — no explanation, no summary, no markdown fences, no backticks.
                   - Use only the tables and columns present in the schema.
                   - Do NOT generate INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements.
                   - Always add a LIMIT clause (max 50 rows) unless the question asks for aggregates.
                   - For product or text searches: NEVER search for the full multi-word phrase as one
                       ILIKE pattern. Instead, split the search into individual meaningful keywords
                       and OR them together across both name and description columns.
                       Example — user asks "wireless headset":
                           WHERE (name ILIKE '%wireless%' OR description ILIKE '%wireless%')
                           OR (name ILIKE '%headset%'  OR description ILIKE '%headset%')
                           OR (name ILIKE '%headphones%' OR description ILIKE '%headphones%')
                       Use your knowledge of synonyms (headset/headphones, laptop/notebook, etc.)
                       to cast a wider net when the exact term may not match.
                  
                   Database schema:
                   {schema}
               """,
            ),
            (
                "human",
                """
                   Question:
                   {question}
               """,
            ),
        ]
    )
    # preprare the chain and invoke with a query
    sql_chain = sql_prompt | llm
    # look for sql query only
    raw_sql = sql_chain.invoke({"schema": schema_info, "question": state["query"]})
    print("========GENERATED raw_sql query is: =====")
    print(raw_sql.content)
    generated_sql = raw_sql.content

    # execute the generated sql query  to get the outout from RDMBS
    try:
        sql_result = db.run(generated_sql)
    except Exception as err:
        sql_result = f"Generated SQL execution error: {err}"

    # connect to LLM to get the natural language response
    structured_llm = llm.with_structured_output(AIResponse)
    nl_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful data analyst. Answer the user's question using
               the SQL query results below. Be concise and format numbers/lists clearly.
               Set policy_citations to empty string,
               page_no to 'N/A', and document_name to 'agentic_rag_db'.
               - Do NOT execute INSERT, UPDATE, DELETE, DROP, or any DML/DDL statements
               even if requested.
               - Politely deny when users are asking for these actions in their queries.
               - Never use tech jargons in your response""",
            ),
            (
                "human",
                "Question: {query}\n\n"
                "SQL Used:\n{sql}\n\n"
                "Query Results:\n{result}",
            ),
        ]
    )

    nl_chain = nl_answer_prompt | structured_llm
    answer = nl_chain.invoke(
        {"query": state["query"], "sql": generated_sql, "result": sql_result}
    )
    print("[nl2sql_node] Answer generated.")
    response = answer.model_dump()
    print("************response in nl2sql********", response)
    response["policy_citations"] = "N/A"
    response["sql_query_executed"] = generated_sql
    # return the sql query is RAGState
    # and also the output in sql_result of RAGState
    # return {
    #     **state,
    #     "generated_sql": generated_sql,
    #     "sql_result": str(sql_result),
    #     "response": response,
    # }
    return {
        "answer": answer.answer,
        "response": response,
        "sql_result": str(sql_result),
        "generated_sql": generated_sql,
    }


def rerank_node(state: RAGState):
    # establish connection with the cohere reranking model
    co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    # send the query and the retrieved_docs to the reranking model

    docs = state["retrieved_docs"]

    print("=======3. INSIDE rerank_node. Before calling reranker =========")

    print("Retrieved docs count:", len(docs))

    if not docs:
        print("No documents found. Skipping reranking.")

        return {**state, "reranked_docs": []}

    rerank_response = co.rerank(
        model="rerank-v3.5",
        query=state["query"],
        documents=[doc.page_content for doc in docs],
        top_n=5,
    )

    # Map Cohere result indices back to LangChain Document objects
    reranked_docs = [docs[r.index] for r in rerank_response.results]

    print(f"[rerank_node] Top {len(reranked_docs)} chunks after reranking:")
    for i, r in enumerate(rerank_response.results):
        print(
            f"  Rank {i+1} | Cohere score: {r.relevance_score:.4f} | original index: {r.index}"
        )

    return {**state, "reranked_docs": reranked_docs}


def generate_answer_node(state: RAGState):
    if not state["reranked_docs"]:

        return {
            **state,
            "response": {
                "answer": "I could not find relevant information.",
                "policy_citations": "",
                "page_no": "",
                "document_name": "",
            },
        }

    llm = _get_llm()
    structured_llm = llm.with_structured_output(AIResponse)

    print("=========4. INSIDE GENERATE ANSWER NODE==========")

    for doc in state["reranked_docs"]:
        print("Metadata: ", doc.metadata)

    # let's prepare the context
    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', -1) + 1 if doc.metadata.get('page') is not None else '?'}]\n{doc.page_content}"
            for doc in state["reranked_docs"]
        ]
    )
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                   You are a helpful credit card assistant. Answer the user's question using only the
                   provided context politely reject if any question asked out of scope
                   don't answer out of this .
                  
                   Citation rules (fill the structured fields):
                   - document_name: comma-separated list of EVERY source document you used.
                   - page_no: comma-separated page numbers, aligned with the documents above.
                   - policy_citations: a readable citation combining each document and its page
                   (e.g.  KB_Credit_Card_Spend_Summarizer.docx, Page 1").
                   - Always cite ALL versions you drew the answer from, not just one.
           """,
            ),
            (
                "human",
                """

                  Conversation history:
                    {history}
                    
                   Context:
                   {context}


                   Question:
                   {query}
               """,
            ),
        ]
    )

    chain = prompt | structured_llm
    result = chain.invoke(
        {"history": history, "context": context, "query": state["query"]}
    )

    # print(f"[generate_answer_node] Answer generated.")
    # return {**state, "response": result.model_dump(),"messages":
    #         [AIMessage(content=result.answer)]}

    return {
        **state,
        "context": context,
        "answer": result.answer,
        "response": result.model_dump(),
        "messages": [AIMessage(content=result.answer)],
    }


def evaluation_node(state: RAGState) -> RAGState:
    print("-------- Evaluating Answer ----------")
    history = "\n".join(
        [f"{msg.type}: {msg.content}" for msg in state.get("messages", [])]
    )
    llm = _get_llm()

    prompt = f"""
           User Preferences: {history}


           Question: {state['query'].lower()}


           Context:  {state.get('context','')} 


           Answer: {state.get('answer','')}

  
           Is the answer is correct and complete based on the context?
           Respond with only: yes or no
        """

    result = llm.invoke(prompt).content.strip()
    attempts = state.get("attempts", 0) + 1
    print("========== EVALUATION RESULT ==========")
    print(result)
    print("Attempt:", attempts)
    print("========================================")
    return {**state, "is_good": result == "yes", "attempts": attempts}


def route(state: RAGState):
    if state["is_good"] or state["attempts"] >= 3:
        return "NO_RETRY_REQUIRED"

    return "RETRY_REQUIRED"


def build_rag_graph(checkpoint):
    workflow = StateGraph(RAGState)
    workflow.add_node("in_memory", add_user_message_node)
    workflow.add_node("intent_check", intent_check_node)
    workflow.add_node("chat_response", chat_response_node)
    workflow.add_node("rephraser", rephraser_node)
    workflow.add_node("router", router_node)
    workflow.add_node("vector_search", vector_search_node)
    workflow.add_node("hybrid_search", hybrid_search_node)
    workflow.add_node("fts", fts_search_node)
    workflow.add_node("nl2sql", nl2sql_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluation", evaluation_node)
    workflow.add_node("image_search", extract_images_node)

    workflow.set_entry_point("in_memory")
    workflow.add_edge("in_memory", "intent_check")
    workflow.add_conditional_edges(
        "intent_check",
        lambda state: state["intent"],
        {"CHITCHAT": "chat_response", "REPHRASER": "rephraser"},
    )
    workflow.add_edge("chat_response", END)
    # workflow.add_edge("in_memory", "rephraser")
    workflow.add_edge("rephraser", "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "FTS": "fts",
            "VECTOR_DB": "vector_search",
            "HYBRID": "hybrid_search",
            "RDBMS": "nl2sql",
            "IMAGE": "image_search",
        },
    )
    # Tools -> Rerank

    workflow.add_edge("vector_search", "rerank")
    workflow.add_edge("fts", "rerank")
    workflow.add_edge("hybrid_search", "rerank")
    workflow.add_edge("rerank", "generate_answer")
    workflow.add_edge("generate_answer", "evaluation")
    workflow.add_conditional_edges(
        "evaluation", route, {"RETRY_REQUIRED": "router", "NO_RETRY_REQUIRED": END}
    )
    workflow.add_edge("evaluation", END)
    workflow.add_edge("image_search", END)

    # checkpoint = InMemorySaver()
    # search_agent = workflow.compile(checkpointer=checkpoint)
    # workflow = build_rag_graph()
    search_agent = workflow.compile(checkpointer=checkpoint)
    # generating and saving the graph visualization
    graph_image = search_agent.get_graph().draw_mermaid_png()
    with open("search_agent.png", "wb") as f:
        f.write(graph_image)

    return search_agent


# rag_graph = build_rag_graph(checkpoint)


async def run_search_agent(rag_graph, query: str, thread_id: str):
    print("============1. INSIDE run_search_agent ")
    initial_state = {
        "query": query,
        # "messages": [],
        "retrieved_docs": [],
        "reranked_docs": [],
        "response": {},
        "images": [],
        "evaluation": {},
        "is_good": False,
        "attempts": 0,
    }
    config = {"configurable": {"thread_id": thread_id}}

    final_state = await rag_graph.ainvoke(initial_state, config=config)
    if final_state.get("route") == "IMAGE":
        print("FINAL STATE:")
        print(final_state)

        print("FINAL IMAGES:")
        print(final_state.get("images"))
        return {"query": query, "images": final_state.get("images", [])}

    state = await rag_graph.get_state(config)

    print("***************am printing my state m,essage ", state.values["messages"])

    return final_state["response"]


# for streaming repsonse:
async def run_search_agent_stream(
    rag_graph,
    query: str,
    thread_id: str,
):

    print("============ INSIDE run_search_agent_stream")

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

    try:

        async for event in rag_graph.astream_events(
            initial_state, config=config, version="v2"
        ):

            kind = event.get("event")

            print("EVENT:", kind)
            print("DATA:", event.get("data"))

            print(kind)

            # Stream LLM tokens
            if kind == "on_chat_model_stream":

                node_name = event.get("metadata", {}).get("langgraph_node")

                print("STREAM NODE:", node_name)

                # only stream final answer nodes
                if node_name not in ["chat_response", "generate_answer"]:
                    continue

                chunk = event["data"]["chunk"]

                content = chunk.content
                if content:

                    yield json.dumps({"content": content}) + "\n"

        # After streaming completes,
        # get final graph state for metadata

        final_state = await rag_graph.aget_state(config)

        state_values = final_state.values

        # send final metadata
        yield {
            "done": True,
            "sources": state_values.get("sources", []),
            "images": state_values.get("images", []),
            "policy_citations": state_values.get("policy_citations", ""),
            "page_no": state_values.get("page_no", ""),
            "document_name": state_values.get("document_name", ""),
        }

    except Exception as e:
        print("Streaming agent error:")
        traceback.print_exc()

        raise e

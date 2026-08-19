from pydantic import BaseModel, Field
from typing import Optional, List


# query api endpoint request format
class QueryRequest(BaseModel):
    query: str = Field(description="The user's question")
    thread_id: str = Field(description="The user's sessionid")


class ImageResult(BaseModel):
    content: str
    source_file: Optional[str] = None
    mime_type: Optional[str] = None
    image_base64: Optional[str] = None


class EvaluationResult(BaseModel):
    faithfulness_score: float = Field(
        description="How well answer is supported by retrieved documents (0-1)"
    )
    relevance_score: float = Field(
        description="How relevant answer is to user query (0-1)"
    )
    completeness_score: float = Field(
        description="Whether answer covers required information (0-1)"
    )
    passed: bool
    feedback: str


# query api endpoint response format
class QueryResponse(BaseModel):
    query: str
    thread_id: str = "user01"
    answer: str
    policy_citations: str
    page_no: str
    document_name: str
    sql_query_executed: Optional[str]
    images: Optional[List[ImageResult]] = []


class AIResponse(BaseModel):
    query: str = Field(description="The given query by user")
    answer: str = Field(description="The generated response")
    policy_citations: str = Field(
        description="Policy citation for the documents retrieved"
    )
    page_no: str = Field(description="Page number in the metadata")
    document_name: str = Field(description="Name of the document")
    sql_query_executed: Optional[str] = Field(
        description="The AI generated and executed SQL query for the query"
    )

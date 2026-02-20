"""Request models for the API."""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="User question to process", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are the termination clauses in the contract?"
            }
        }


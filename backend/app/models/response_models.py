"""Response models for the API."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievedClause(BaseModel):
    """Model for a retrieved clause from Pinecone."""
    text: str = Field(..., description="Clause text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Clause metadata")
    score: float = Field(..., description="Similarity score")


class EvaluationResult(BaseModel):
    """Model for FRAMES evaluation results."""
    factuality: float = Field(..., description="Factuality score (0-1)", ge=0.0, le=1.0)
    completeness: float = Field(..., description="Completeness score (0-1)", ge=0.0, le=1.0)
    reliability: float = Field(..., description="Reliability score (0-1)", ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, description="Evaluation reasoning")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    final_answer: str = Field(..., description="Final answer after processing")
    retrieved_clauses: List[RetrievedClause] = Field(default_factory=list, description="Retrieved clauses from Pinecone")
    draft_answer: Optional[str] = Field(None, description="Initial draft answer before correction")
    evaluation: Optional[EvaluationResult] = Field(None, description="FRAMES evaluation results")
    self_corrected: bool = Field(False, description="Whether self-correction was applied")
    alert_triggered: bool = Field(False, description="Whether an alert email was sent")


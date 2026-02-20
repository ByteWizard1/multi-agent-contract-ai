"""Tests for data models."""
import pytest

from app.models.query_models import QueryRequest
from app.models.response_models import EvaluationResult, QueryResponse, RetrievedClause


def test_query_request():
    """Test QueryRequest model."""
    request = QueryRequest(question="Test question")
    assert request.question == "Test question"


def test_retrieved_clause():
    """Test RetrievedClause model."""
    clause = RetrievedClause(
        text="Test clause",
        metadata={"key": "value"},
        score=0.95,
    )
    assert clause.text == "Test clause"
    assert clause.score == 0.95
    assert clause.metadata == {"key": "value"}


def test_evaluation_result():
    """Test EvaluationResult model."""
    eval_result = EvaluationResult(
        factuality=0.8,
        completeness=0.9,
        reliability=0.85,
        reasoning="Test reasoning",
    )
    assert eval_result.factuality == 0.8
    assert eval_result.completeness == 0.9
    assert eval_result.reliability == 0.85
    assert eval_result.reasoning == "Test reasoning"


def test_query_response():
    """Test QueryResponse model."""
    response = QueryResponse(
        final_answer="Test answer",
        retrieved_clauses=[],
        draft_answer="Draft answer",
        self_corrected=False,
        alert_triggered=False,
    )
    assert response.final_answer == "Test answer"
    assert response.self_corrected is False
    assert response.alert_triggered is False


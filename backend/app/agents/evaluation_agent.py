"""Evaluation agent using FRAMES method."""
from typing import Dict, List

from app.models.response_models import EvaluationResult, RetrievedClause
from app.services.bedrock_client import BedrockClient
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationAgent:
    """Agent responsible for FRAMES evaluation (Factuality, Retrieval, Reasoning, Evaluation, Self-correction)."""
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize evaluation agent.
        
        Args:
            bedrock_client: Bedrock client for LLM evaluation
        """
        self.bedrock_client = bedrock_client
        logger.info("EvaluationAgent initialized")
    
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_clauses: List[RetrievedClause],
    ) -> EvaluationResult:
        """
        Evaluate answer using FRAMES method.
        
        FRAMES evaluates:
        - Factuality: How well the answer aligns with retrieved evidence
        - Completeness: Whether the answer addresses all aspects of the query
        - Reliability: Overall confidence in the answer
        
        Args:
            query: Original user query
            answer: Generated answer to evaluate
            retrieved_clauses: Evidence clauses used to generate the answer
            
        Returns:
            EvaluationResult with scores and reasoning
        """
        logger.info("Starting FRAMES evaluation", query=query, answer_length=len(answer))
        
        try:
            # Build evidence context
            evidence_parts = []
            for i, clause in enumerate(retrieved_clauses, 1):
                evidence_parts.append(f"[Evidence {i}] {clause.text}")
            evidence = "\n\n".join(evidence_parts)
            
            # FRAMES evaluation prompt
            evaluation_prompt = f"""You are an expert evaluator using the FRAMES methodology to assess answer quality.

Question: {query}

Answer to Evaluate:
{answer}

Supporting Evidence:
{evidence}

Please evaluate the answer across three dimensions (provide scores from 0.0 to 1.0):

1. FACTUALITY: How well does the answer align with the provided evidence? Does it accurately represent the information in the clauses?
   - 1.0: Perfect alignment, all claims are directly supported
   - 0.7-0.9: Mostly accurate with minor gaps
   - 0.4-0.6: Partially accurate, some unsupported claims
   - 0.0-0.3: Significant misalignment with evidence

2. COMPLETENESS: Does the answer address all aspects of the question? Are there important details missing?
   - 1.0: Comprehensive, addresses all aspects
   - 0.7-0.9: Mostly complete, minor gaps
   - 0.4-0.6: Partially complete, missing some aspects
   - 0.0-0.3: Incomplete, missing major aspects

3. RELIABILITY: Overall confidence in the answer's correctness and usefulness?
   - 1.0: Highly reliable, well-supported
   - 0.7-0.9: Generally reliable
   - 0.4-0.6: Moderately reliable, some concerns
   - 0.0-0.3: Low reliability

Respond in JSON format:
{{
    "factuality": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "reliability": <float 0.0-1.0>,
    "reasoning": "<brief explanation of scores>"
}}"""
            
            # Get evaluation from Claude
            evaluation_text = self.bedrock_client.invoke_claude(
                prompt=evaluation_prompt,
                system_prompt="You are a precise evaluator. Always respond with valid JSON only.",
                temperature=0.2,  # Low temperature for consistent evaluation
            )
            
            # Parse JSON response
            import json
            try:
                # Extract JSON from response (handle markdown code blocks if present)
                evaluation_text_clean = evaluation_text.strip()
                if evaluation_text_clean.startswith("```json"):
                    evaluation_text_clean = evaluation_text_clean[7:]
                if evaluation_text_clean.startswith("```"):
                    evaluation_text_clean = evaluation_text_clean[3:]
                if evaluation_text_clean.endswith("```"):
                    evaluation_text_clean = evaluation_text_clean[:-3]
                evaluation_text_clean = evaluation_text_clean.strip()
                
                eval_dict = json.loads(evaluation_text_clean)
                
                result = EvaluationResult(
                    factuality=float(eval_dict.get("factuality", 0.5)),
                    completeness=float(eval_dict.get("completeness", 0.5)),
                    reliability=float(eval_dict.get("reliability", 0.5)),
                    reasoning=eval_dict.get("reasoning", ""),
                )
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning("Failed to parse evaluation JSON, using default scores", error=str(e))
                result = EvaluationResult(
                    factuality=0.5,
                    completeness=0.5,
                    reliability=0.5,
                    reasoning="Error parsing evaluation response",
                )
            
            logger.info(
                "Evaluation completed",
                factuality=result.factuality,
                completeness=result.completeness,
                reliability=result.reliability,
            )
            
            return result
            
        except Exception as e:
            logger.error("Error in evaluation agent", error=str(e))
            # Return default evaluation on error
            return EvaluationResult(
                factuality=0.5,
                completeness=0.5,
                reliability=0.5,
                reasoning=f"Evaluation error: {str(e)}",
            )
    
    def should_correct(self, evaluation: EvaluationResult) -> bool:
        """
        Determine if self-correction is needed based on evaluation.
        
        Args:
            evaluation: Evaluation result
            
        Returns:
            True if correction is needed, False otherwise
        """
        needs_correction = evaluation.factuality < settings.FACTUALITY_THRESHOLD
        logger.info(
            "Checking if correction needed",
            factuality=evaluation.factuality,
            threshold=settings.FACTUALITY_THRESHOLD,
            needs_correction=needs_correction,
        )
        return needs_correction


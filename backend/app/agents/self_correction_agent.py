"""Self-correction agent for improving answers based on evaluation."""
from typing import Dict, List

from app.models.response_models import RetrievedClause
from app.services.bedrock_client import BedrockClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SelfCorrectionAgent:
    """Agent responsible for self-correcting answers based on evaluation feedback."""
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize self-correction agent.
        
        Args:
            bedrock_client: Bedrock client for LLM inference
        """
        self.bedrock_client = bedrock_client
        logger.info("SelfCorrectionAgent initialized")
    
    def correct(
        self,
        query: str,
        original_answer: str,
        retrieved_clauses: List[RetrievedClause],
        evaluation_reasoning: str,
    ) -> str:
        """
        Generate a corrected answer based on evaluation feedback.
        
        Args:
            query: Original user query
            original_answer: Initial answer that needs correction
            retrieved_clauses: Evidence clauses
            evaluation_reasoning: Reasoning from evaluation about what needs improvement
            
        Returns:
            Corrected answer
        """
        logger.info("Starting self-correction", query=query, evaluation_reasoning=evaluation_reasoning)
        
        try:
            # Build evidence context
            context_parts = []
            for i, clause in enumerate(retrieved_clauses, 1):
                clause_text = clause.text
                metadata_info = ""
                if clause.metadata:
                    metadata_info = f" (Metadata: {clause.metadata})"
                context_parts.append(f"[Clause {i}] {clause_text}{metadata_info}")
            
            context = "\n\n".join(context_parts)
            
            # Construct correction prompt
            system_prompt = """You are an expert legal analyst assistant. Your role is to improve and correct answers based on evaluation feedback, ensuring strict alignment with the provided evidence."""
            
            correction_prompt = f"""Question: {query}

Original Answer (needs improvement):
{original_answer}

Evaluation Feedback:
{evaluation_reasoning}

Relevant Contract Clauses:
{context}

Please generate an improved, corrected answer that:
1. Strictly adheres to the evidence in the clauses
2. Addresses the issues identified in the evaluation feedback
3. Maintains accuracy and completeness
4. Is well-structured and clear

Provide only the corrected answer, without additional commentary."""
            
            # Generate corrected answer
            corrected_answer = self.bedrock_client.invoke_claude(
                prompt=correction_prompt,
                system_prompt=system_prompt,
                temperature=0.2,  # Low temperature for more factual corrections
            )
            
            logger.info(
                "Self-correction completed",
                query=query,
                original_length=len(original_answer),
                corrected_length=len(corrected_answer),
            )
            
            return corrected_answer.strip()
            
        except Exception as e:
            logger.error("Error in self-correction agent", error=str(e), query=query)
            # Return original answer if correction fails
            logger.warning("Self-correction failed, returning original answer")
            return original_answer


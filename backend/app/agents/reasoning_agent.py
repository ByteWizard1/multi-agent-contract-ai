
from typing import Dict, List

from langchain.tools import Tool

from app.models.response_models import RetrievedClause
from app.services.bedrock_client import BedrockClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningAgent:
    """Agent responsible for reasoning over retrieved clauses to generate answers."""
    
    def __init__(self, bedrock_client: BedrockClient):
        """
        Initialize reasoning agent.
        
        Args:
            bedrock_client: Bedrock client for LLM inference
        """
        self.bedrock_client = bedrock_client
        logger.info("ReasoningAgent initialized")
    
    def reason(
        self,
        query: str,
        retrieved_clauses: List[RetrievedClause],
    ) -> Dict[str, str]:
        """
        Generate an answer by reasoning over retrieved clauses.
        
        Args:
            query: User query
            retrieved_clauses: List of retrieved clauses with metadata
            
        Returns:
            Dictionary with 'answer' and 'context' keys
        """
        logger.info("Starting reasoning", query=query, clauses_count=len(retrieved_clauses))
        
        try:
            # Build context from retrieved clauses
            context_parts = []
            for i, clause in enumerate(retrieved_clauses, 1):
                clause_text = clause.text
                metadata_info = ""
                if clause.metadata:
                    metadata_info = f" (Metadata: {clause.metadata})"
                context_parts.append(f"[Clause {i}] {clause_text}{metadata_info}")
            
            context = "\n\n".join(context_parts)
            
            # Construct system prompt
            system_prompt = """You are an expert legal analyst assistant. Your role is to analyze contract clauses and provide accurate, comprehensive answers based on the retrieved evidence.

Guidelines:
- Base your answer strictly on the provided clauses and their metadata
- Synthesize information from multiple clauses when relevant
- Be precise and factual
- If the clauses don't contain enough information, clearly state what is missing
- Structure your answer logically and clearly
- Include relevant details from metadata when helpful"""
            
            # Construct user prompt
            user_prompt = f"""Question: {query}

Relevant Contract Clauses:
{context}

Please provide a comprehensive answer to the question based on the retrieved clauses above. Include specific details and references where applicable."""
            
            # Generate answer using Claude
            answer = self.bedrock_client.invoke_claude(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,  # Lower temperature for more factual responses
            )
            
            result = {
                "answer": answer,
                "context": context,
            }
            
            logger.info(
                "Reasoning completed",
                query=query,
                answer_length=len(answer),
                context_length=len(context),
            )
            
            return result
            
        except Exception as e:
            logger.error("Error in reasoning agent", error=str(e), query=query)
            raise
    
    def get_tool(self) -> Tool:
        """Get LangChain tool for reasoning agent."""
        return Tool(
            name="reasoning",
            description="Generates an answer by reasoning over retrieved contract clauses using Claude Sonnet",
            func=self.reason,
        )


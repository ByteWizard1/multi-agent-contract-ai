"""Retrieval agent for fetching relevant clauses from Pinecone."""
from typing import List

from langchain.tools import Tool

from app.models.response_models import RetrievedClause
from app.services.bedrock_client import BedrockClient
from app.services.pinecone_client import PineconeClient
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalAgent:
    """Agent responsible for retrieving relevant clauses from Pinecone."""
    
    def __init__(self, bedrock_client: BedrockClient, pinecone_client: PineconeClient):
        """
        Initialize retrieval agent.
        
        Args:
            bedrock_client: Bedrock client for embeddings
            pinecone_client: Pinecone client for vector search
        """
        self.bedrock_client = bedrock_client
        self.pinecone_client = pinecone_client
        logger.info("RetrievalAgent initialized")
    
    def retrieve(self, query: str, top_k: int = None) -> List[RetrievedClause]:
        """
        Retrieve relevant clauses for a given query.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve (defaults to config value)
            
        Returns:
            List of retrieved clauses with metadata and scores
        """
        if top_k is None:
            top_k = settings.TOP_K_CLAUSES
        
        logger.info("Starting retrieval", query=query, top_k=top_k)
        
        try:
            # Generate embedding for the query
            query_embedding = self.bedrock_client.generate_embedding(query)
            
            # Search Pinecone
            clauses = self.pinecone_client.search(
                query_vector=query_embedding,
                top_k=top_k,
            )
            
            logger.info(
                "Retrieval completed",
                query=query,
                clauses_retrieved=len(clauses),
            )
            
            return clauses
            
        except Exception as e:
            logger.error("Error in retrieval agent", error=str(e), query=query)
            raise
    
    def get_tool(self) -> Tool:
        """Get LangChain tool for retrieval agent."""
        return Tool(
            name="retrieval",
            description="Retrieves relevant contract clauses from the vector database based on a query",
            func=self.retrieve,
        )


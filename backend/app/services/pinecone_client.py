"""Pinecone vector database client."""
from typing import List, Optional

from pinecone import Pinecone
from pinecone.exceptions import PineconeException

from app.models.response_models import RetrievedClause
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PineconeClient:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self):
        """Initialize Pinecone client."""
        try:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
            self.index = self.pc.Index(settings.PINECONE_INDEX_NAME)
            logger.info("Pinecone client initialized", index_name=settings.PINECONE_INDEX_NAME)
        except PineconeException as e:
            logger.error("Error initializing Pinecone client", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error initializing Pinecone", error=str(e))
            raise
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[dict] = None,
        include_metadata: bool = True,
    ) -> List[RetrievedClause]:
        """
        Search for similar clauses in Pinecone.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter
            include_metadata: Whether to include metadata
            
        Returns:
            List of retrieved clauses with scores
        """
        try:
            query_response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter,
                include_metadata=include_metadata,
            )
            
            clauses = []
            for match in query_response.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", metadata.get("clause_text", ""))
                
                clauses.append(
                    RetrievedClause(
                        text=text,
                        metadata=metadata,
                        score=float(match.get("score", 0.0)),
                    )
                )
            
            logger.info(
                "Pinecone search completed",
                query_dim=len(query_vector),
                top_k=top_k,
                results_count=len(clauses),
            )
            
            return clauses
            
        except PineconeException as e:
            logger.error("Error searching Pinecone", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error searching Pinecone", error=str(e))
            raise


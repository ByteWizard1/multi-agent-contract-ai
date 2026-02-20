"""Configuration settings for the agentic AI backend."""
import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = "contract-clauses-dataset"
    
    # Bedrock Model IDs
    BEDROCK_LLM_MODEL_ID: str = "anthropic.claude-3-haiku-20240307-v1:0"
    BEDROCK_EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v1"
    
    # Email Configuration
    SES_FROM_EMAIL: str = "bhgshrchr@gmail.com"
    
    # Evaluation Thresholds
    FACTUALITY_THRESHOLD: float = 0.7
    
    # Retrieval Settings
    TOP_K_CLAUSES: int = 5
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


# Global settings instance
settings = Settings()


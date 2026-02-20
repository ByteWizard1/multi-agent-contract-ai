"""AWS Bedrock client for LLM and embeddings."""
import json
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BedrockClient:
    """Client for interacting with AWS Bedrock."""
    
    def __init__(self):
        """Initialize Bedrock client."""
        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        logger.info("Bedrock client initialized", region=settings.AWS_REGION)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Titan Text Embeddings V2.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            body = json.dumps({"inputText": text})
            
            response = self.bedrock_runtime.invoke_model(
                modelId=settings.BEDROCK_EMBEDDING_MODEL_ID,
                body=body,
            )
            
            response_body = json.loads(response["body"].read())
            embedding = response_body.get("embedding", [])
            
            logger.info("Embedding generated", text_length=len(text), embedding_dim=len(embedding))
            return embedding
            
        except ClientError as e:
            logger.error("Error generating embedding", error=str(e), error_code=e.response.get("Error", {}).get("Code"))
            raise
        except Exception as e:
            logger.error("Unexpected error generating embedding", error=str(e))
            raise
    
    def invoke_claude(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        """
        Invoke Claude 3.5 Sonnet via Bedrock.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            
            body: Dict[str, Any] = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            
            if system_prompt:
                body["system"] = system_prompt
            
            response = self.bedrock_runtime.invoke_model(
                modelId=settings.BEDROCK_LLM_MODEL_ID,
                body=json.dumps(body),
            )
            
            response_body = json.loads(response["body"].read())
            
            # Extract content from Claude's response
            content = ""
            for content_block in response_body.get("content", []):
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")
            
            logger.info(
                "Claude response generated",
                prompt_length=len(prompt),
                response_length=len(content),
                tokens_used=response_body.get("usage", {}).get("input_tokens", 0),
            )
            
            return content.strip()
            
        except ClientError as e:
            logger.error("Error invoking Claude", error=str(e), error_code=e.response.get("Error", {}).get("Code"))
            raise
        except Exception as e:
            logger.error("Unexpected error invoking Claude", error=str(e))
            raise


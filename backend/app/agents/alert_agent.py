"""Alert agent for sending email notifications."""
from typing import Optional

from app.services.ses_client import SESClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class AlertAgent:
    """Agent responsible for sending alert emails when critical insights are detected."""
    
    def __init__(self, ses_client: SESClient):
        """
        Initialize alert agent.
        
        Args:
            ses_client: SES client for sending emails
        """
        self.ses_client = ses_client
        logger.info("AlertAgent initialized")
    
    def should_alert(self, query: str, answer: str) -> bool:
        """
        Determine if an alert should be triggered based on query/answer content.
        
        Looks for keywords indicating risks, compliance issues, or critical insights.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            True if alert should be sent, False otherwise
        """
        # Keywords that might indicate critical issues
        risk_keywords = [
            "termination",
            "breach",
            "penalty",
            "violation",
            "non-compliance",
            "risk",
            "liability",
            "damages",
            "critical",
            "urgent",
            "important",
            "deadline",
            "expiration",
        ]
        
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Check if query or answer contains risk keywords
        has_risk_keyword = any(keyword in query_lower or keyword in answer_lower for keyword in risk_keywords)
        
        logger.info(
            "Checking alert condition",
            query=query,
            has_risk_keyword=has_risk_keyword,
        )
        
        return has_risk_keyword
    
    def send_alert(
        self,
        query: str,
        answer: str,
        recipient: Optional[str] = None,
    ) -> bool:
        """
        Send an alert email if conditions are met.
        
        Args:
            query: Original user query
            answer: Generated answer
            recipient: Optional recipient email (defaults to configured email)
            
        Returns:
            True if alert was sent, False otherwise
        """
        if not self.should_alert(query, answer):
            logger.info("Alert conditions not met, skipping email")
            return False
        
        try:
            subject = f"AI System Alert: Critical Contract Insight Detected"
            
            body = f"""An AI system alert has been triggered based on the following query and response:

QUERY:
{query}

RESPONSE:
{answer}

Please review this information for potential risks, compliance issues, or critical insights.

---
This is an automated alert from the Agentic AI System.
"""
            
            success = self.ses_client.send_alert_email(
                subject=subject,
                body=body,
                recipient=recipient,
            )
            
            logger.info("Alert processing completed", success=success, query=query)
            return success
            
        except Exception as e:
            logger.error("Error in alert agent", error=str(e), query=query)
            return False


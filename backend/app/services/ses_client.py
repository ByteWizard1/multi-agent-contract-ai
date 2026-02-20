"""AWS SES client for sending emails."""
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SESClient:
    """Client for sending emails via AWS SES."""
    
    def __init__(self):
        """Initialize SES client."""
        self.ses_client = boto3.client(
            "ses",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        logger.info("SES client initialized", region=settings.AWS_REGION, from_email=settings.SES_FROM_EMAIL)
    
    def send_alert_email(
        self,
        subject: str,
        body: str,
        recipient: Optional[str] = None,
    ) -> bool:
        """
        Send an alert email via AWS SES.
        
        Args:
            subject: Email subject
            body: Email body (HTML or plain text)
            recipient: Recipient email address (defaults to from_email)
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not recipient:
            recipient = settings.SES_FROM_EMAIL
        
        try:
            response = self.ses_client.send_email(
                Source=settings.SES_FROM_EMAIL,
                Destination={"ToAddresses": [recipient]},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
                },
            )
            
            message_id = response.get("MessageId", "")
            logger.info(
                "Alert email sent successfully",
                message_id=message_id,
                recipient=recipient,
                subject=subject,
            )
            
            return True
            
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            logger.error(
                "Error sending alert email",
                error=str(e),
                error_code=error_code,
                recipient=recipient,
            )
            return False
        except Exception as e:
            logger.error("Unexpected error sending alert email", error=str(e))
            return False


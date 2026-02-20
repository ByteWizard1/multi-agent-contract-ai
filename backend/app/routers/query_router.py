"""Query processing router."""
from fastapi import APIRouter, HTTPException

from app.agents.alert_agent import AlertAgent
from app.agents.evaluation_agent import EvaluationAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.self_correction_agent import SelfCorrectionAgent
from app.models.query_models import QueryRequest
from app.models.response_models import QueryResponse
from app.services.bedrock_client import BedrockClient
from app.services.pinecone_client import PineconeClient
from app.services.ses_client import SESClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


# Initialize clients (in production, use dependency injection)
_bedrock_client: BedrockClient = None
_pinecone_client: PineconeClient = None
_ses_client: SESClient = None


def get_clients():
    """Get or initialize service clients."""
    global _bedrock_client, _pinecone_client, _ses_client
    
    if _bedrock_client is None:
        _bedrock_client = BedrockClient()
    if _pinecone_client is None:
        _pinecone_client = PineconeClient()
    if _ses_client is None:
        _ses_client = SESClient()
    
    return _bedrock_client, _pinecone_client, _ses_client


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a user query through the full agent pipeline.
    
    Pipeline: Retrieval → Reasoning → Evaluation → (optional) Self-Correction → (optional) Alert
    """
    logger.info("Processing query", question=request.question)
    
    try:
        # Initialize clients
        bedrock_client, pinecone_client, ses_client = get_clients()
        
        # Initialize agents
        retrieval_agent = RetrievalAgent(bedrock_client, pinecone_client)
        reasoning_agent = ReasoningAgent(bedrock_client)
        evaluation_agent = EvaluationAgent(bedrock_client)
        self_correction_agent = SelfCorrectionAgent(bedrock_client)
        alert_agent = AlertAgent(ses_client)
        
        # Step 1: Retrieval
        retrieved_clauses = retrieval_agent.retrieve(request.question)
        
        if not retrieved_clauses:
            logger.warning("No clauses retrieved", question=request.question)
            return QueryResponse(
                final_answer="I couldn't find any relevant contract clauses for your query. Please try rephrasing your question.",
                retrieved_clauses=[],
            )
        
        # Step 2: Reasoning
        reasoning_result = reasoning_agent.reason(request.question, retrieved_clauses)
        draft_answer = reasoning_result["answer"]
        final_answer = draft_answer
        self_corrected = False
        
        # Step 3: Evaluation
        evaluation = evaluation_agent.evaluate(
            query=request.question,
            answer=draft_answer,
            retrieved_clauses=retrieved_clauses,
        )
        
        # Step 4: Self-Correction (if needed)
        if evaluation_agent.should_correct(evaluation):
            logger.info("Self-correction triggered", factuality=evaluation.factuality)
            final_answer = self_correction_agent.correct(
                query=request.question,
                original_answer=draft_answer,
                retrieved_clauses=retrieved_clauses,
                evaluation_reasoning=evaluation.reasoning or "Factuality below threshold",
            )
            self_corrected = True
        
        # Step 5: Alert (if needed)
        alert_triggered = alert_agent.send_alert(request.question, final_answer)
        
        # Build response
        response = QueryResponse(
            final_answer=final_answer,
            retrieved_clauses=retrieved_clauses,
            draft_answer=draft_answer,
            evaluation=evaluation,
            self_corrected=self_corrected,
            alert_triggered=alert_triggered,
        )
        
        logger.info(
            "Query processing completed",
            question=request.question,
            self_corrected=self_corrected,
            alert_triggered=alert_triggered,
        )
        
        return response
        
    except Exception as e:
        logger.error("Error processing query", error=str(e), question=request.question)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


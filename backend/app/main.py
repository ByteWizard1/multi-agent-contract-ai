"""Main FastAPI application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health_router, query_router
from app.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Agentic AI System",
    description="AI backend with retrieval, reasoning, evaluation, and self-correction capabilities",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router.router, tags=["health"])
app.include_router(query_router.router, tags=["query"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Application starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Application shutting down")


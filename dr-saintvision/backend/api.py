"""
FastAPI Backend for DR-Saintvision
RESTful API for the multi-agent debate system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import logging
import uuid

from models.debate_manager import DebateManager, DebateConfig, DebateStatus
from backend.database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DR-Saintvision API",
    description="Multi-Agent AI Debate System for Enhanced Reasoning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = DatabaseManager()
debate_manager = None


# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The question to analyze")
    user_id: Optional[str] = Field(default="anonymous", description="User identifier")
    use_ollama: bool = Field(default=False, description="Use Ollama for model inference")
    parallel: bool = Field(default=True, description="Run search and reasoning in parallel")


class QueryResponse(BaseModel):
    query_id: str
    query: str
    status: str
    final_answer: Optional[str] = None
    confidence_scores: Optional[Dict[str, float]] = None
    debate_time: Optional[float] = None
    timestamp: datetime


class DebateDetailResponse(BaseModel):
    query_id: str
    query: str
    search_analysis: Dict[str, Any]
    reasoning_analysis: Dict[str, Any]
    final_synthesis: Dict[str, Any]
    confidence_scores: Dict[str, float]
    debate_time: float
    status: str
    timestamp: datetime


class FeedbackRequest(BaseModel):
    debate_id: int
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(default="", max_length=1000)


class StatisticsResponse(BaseModel):
    total_debates: int
    completed_debates: int
    failed_debates: int
    success_rate: float
    average_time: float
    average_confidence: float


class HistoryItem(BaseModel):
    query_id: str
    query: str
    confidence_scores: Optional[Dict[str, float]]
    debate_time: Optional[float]
    status: str
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize debate manager on startup"""
    global debate_manager
    logger.info("Starting DR-Saintvision API...")
    # Debate manager will be initialized on first request with specific config


def get_debate_manager(use_ollama: bool = False, parallel: bool = True) -> DebateManager:
    """Get or create debate manager with specified config"""
    global debate_manager

    config = DebateConfig(
        use_ollama=use_ollama,
        parallel_initial=parallel
    )

    # Create new manager if config changed or not initialized
    if debate_manager is None:
        debate_manager = DebateManager(config=config)

    return debate_manager


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DR-Saintvision API",
        "version": "1.0.0",
        "description": "Multi-Agent AI Debate System",
        "endpoints": {
            "analyze": "/analyze",
            "history": "/history/{user_id}",
            "debate": "/debate/{query_id}",
            "stats": "/stats",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database": "connected"
    }


@app.post("/analyze", response_model=QueryResponse)
async def analyze_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Analyze a query using the multi-agent debate system

    This endpoint initiates a debate between three AI agents:
    - Search Agent (Mistral): Web search and RAG
    - Reasoning Agent (Llama): Deep logical reasoning
    - Synthesis Agent (Qwen): Final synthesis and judgment
    """
    query_id = f"debate_{uuid.uuid4().hex[:12]}"

    try:
        # Get debate manager with requested config
        manager = get_debate_manager(
            use_ollama=request.use_ollama,
            parallel=request.parallel
        )

        # Conduct debate
        result = await manager.conduct_debate(request.query)

        # Save to database
        db.save_debate(
            query_id=query_id,
            query=request.query,
            search_result=result.search_analysis,
            reasoning_result=result.reasoning_analysis,
            synthesis_result=result.final_synthesis,
            confidence_scores=result.confidence_scores,
            debate_time=result.debate_time,
            status=result.status.value,
            user_id=request.user_id
        )

        return QueryResponse(
            query_id=query_id,
            query=request.query,
            status=result.status.value,
            final_answer=result.final_synthesis.get('final_answer'),
            confidence_scores=result.confidence_scores,
            debate_time=result.debate_time,
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debate/{query_id}", response_model=DebateDetailResponse)
async def get_debate_detail(query_id: str):
    """Get detailed results for a specific debate"""
    debate = db.get_debate(query_id)

    if not debate:
        raise HTTPException(status_code=404, detail="Debate not found")

    return DebateDetailResponse(
        query_id=debate['query_id'],
        query=debate['query'],
        search_analysis=debate['search_result'] or {},
        reasoning_analysis=debate['reasoning_result'] or {},
        final_synthesis=debate['synthesis_result'] or {},
        confidence_scores=debate['confidence_scores'] or {},
        debate_time=debate['debate_time'] or 0,
        status=debate['status'],
        timestamp=debate['timestamp']
    )


@app.get("/history/{user_id}")
async def get_user_history(
    user_id: str,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0)
):
    """Get debate history for a user"""
    history = db.get_user_history(user_id, limit=limit, offset=offset)

    return {
        "user_id": user_id,
        "count": len(history),
        "history": history
    }


@app.get("/recent")
async def get_recent_debates(limit: int = Query(default=20, le=50)):
    """Get most recent debates"""
    debates = db.get_recent_debates(limit=limit)
    return {
        "count": len(debates),
        "debates": debates
    }


@app.get("/stats", response_model=StatisticsResponse)
async def get_statistics():
    """Get system statistics"""
    stats = db.get_statistics()
    return StatisticsResponse(**stats)


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit feedback for a debate"""
    try:
        db.save_feedback(
            debate_id=request.debate_id,
            rating=request.rating,
            comment=request.comment
        )
        return {"status": "success", "message": "Feedback submitted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_debates(
    keyword: str = Query(..., min_length=1),
    limit: int = Query(default=20, le=50)
):
    """Search debates by keyword"""
    results = db.search_debates(keyword, limit=limit)
    return {
        "keyword": keyword,
        "count": len(results),
        "results": results
    }


@app.delete("/debate/{query_id}")
async def delete_debate(query_id: str):
    """Delete a specific debate"""
    success = db.delete_debate(query_id)
    if success:
        return {"status": "success", "message": f"Debate {query_id} deleted"}
    raise HTTPException(status_code=404, detail="Debate not found")


@app.post("/quick")
async def quick_analyze(query: str = Query(..., min_length=1, max_length=500)):
    """Quick analysis endpoint - returns just the final answer"""
    try:
        manager = get_debate_manager(use_ollama=False)
        answer = await manager.quick_debate(query)
        return {
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare")
async def compare_approaches(query: str = Query(..., min_length=1)):
    """Compare single agent vs multi-agent debate"""
    try:
        manager = get_debate_manager()
        comparison = await manager.compare_single_vs_debate(query)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

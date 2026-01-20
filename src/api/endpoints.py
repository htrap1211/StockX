from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from pydantic import BaseModel
import asyncio
from src.pipeline_enhanced import EnhancedRecommendationPipeline, RecommendationPipeline

router = APIRouter()

# Enhanced Schema with rank info
class RecommendationResponse(BaseModel):
    symbol: str
    market: str
    recommendation: str
    confidence_score: str
    time_horizon: str
    entry_price: float
    stop_loss: float
    reasoning: List[str]
    analysis_date: str
    rank_percentile: Optional[str] = None
    rank_signal: Optional[str] = None

# Dependency
def get_enhanced_pipeline():
    return EnhancedRecommendationPipeline()

def get_pipeline():
    return RecommendationPipeline()

@router.get("/recommendations", response_model=List[RecommendationResponse])
async def get_recommendations(
    market: str = Query(..., regex="^(US|IN)$"), 
    symbols: Optional[str] = Query(None, description="Comma separated list of symbols"),
    pipeline: EnhancedRecommendationPipeline = Depends(get_enhanced_pipeline)
):
    """
    Get cross-sectionally ranked recommendations.
    
    NEW: Uses regime detection and percentile ranking.
    """
    if not symbols:
        # Default watchlists for MVP
        if market == 'US':
            target_symbols = ["AAPL", "NVDA", "TSLA", "AMD", "MSFT", "GOOGL", "META", "AMZN"]
        else:
            target_symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    else:
        target_symbols = [s.strip() for s in symbols.split(",")]
    
    # Use enhanced pipeline for universe ranking
    results = await pipeline.get_universe_recommendations(target_symbols, market)
    
    # Convert to response format
    clean_results = []
    for res in results:
        if "error" in res:
            continue
        
        # Map rank_signal to recommendation
        recommendation = res.get('rank_signal', 
                                res.get('recommendation', 'WATCH'))
        
        clean_results.append({
            "symbol": res['symbol'],
            "market": res['market'],
            "recommendation": recommendation,
            "confidence_score": res['confidence_score'],
            "time_horizon": res['time_horizon'],
            "entry_price": res['entry_price'],
            "stop_loss": res['stop_loss'],
            "reasoning": res['reasoning'],
            "analysis_date": res['analysis_date'],
            "rank_percentile": res.get('rank_percentile'),
            "rank_signal": res.get('rank_signal')
        })

        
    # Add disclaimer to all
    for res in clean_results:
        res["reasoning"].append("DISCLAIMER: AI-generated for educational use only. Not financial advice.")

    return clean_results

@router.get("/stock/{symbol}")
async def get_stock_analysis(
    symbol: str, 
    market: str = "US",
    pipeline: RecommendationPipeline = Depends(get_pipeline)
):
    """
    Get detailed analysis for a specific stock.
    """
    res = await pipeline.get_recommendation(symbol, market)
    if "error" in res:
        raise HTTPException(status_code=404, detail=res["error"])
    return res

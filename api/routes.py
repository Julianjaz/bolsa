import datetime
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from market.data import YahooFinanceProvider, get_market_state_as_of
from market.indicators import add_technical_indicators
from market.support_resistance import detect_swing_pivots
from market.patterns import detect_patterns
from strategy.technical_score import append_technical_score
from strategy.risk_management import append_risk_management
from events.earnings import get_earnings_context
from market.regime import get_market_regime
from llm.gemini_client import GeminiClient
from strategy.decision_engine import generate_decision

router = APIRouter()

class AnalyzeRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"
    analysis_date: Optional[str] = None
    use_gemini: bool = False

@router.get("/health")
def health_check():
    return {"status": "ok", "service": "Swing Trading API"}

@router.post("/analyze")
def analyze_symbol(request: AnalyzeRequest):
    provider = YahooFinanceProvider()
    
    # Si no hay fecha, asume el día actual
    if not request.analysis_date:
        request.analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    df = get_market_state_as_of(provider, request.symbol, request.analysis_date, lookback_days=365)
    
    if df.empty or len(df) < 60:
        raise HTTPException(status_code=400, detail=f"Insufficient historical data for {request.symbol}")
        
    # Calcular indicadores
    df = add_technical_indicators(df)
    df = detect_swing_pivots(df)
    df = detect_patterns(df)
    df = append_technical_score(df)
    df = append_risk_management(df)
    
    last_row = df.iloc[-1]
    
    # Recolectar estado
    analysis_state = {
        "ticker": request.symbol,
        "analysis_date": request.analysis_date,
        "timeframe": request.timeframe,
        "technical": {
            "close": float(last_row['Close']),
            "EMA_20_gt_EMA_50": bool(last_row.get('EMA_20_gt_EMA_50', False)),
            "RSI": float(last_row.get('RSI_14', 0)),
            "Setup": str(last_row.get('Setup', 'NONE')),
            "Score_Is_Long": bool(last_row.get('Score_Is_Long', True))
        },
        "technical_score": float(last_row.get('Technical_Score', 0)),
        "risk_management": {
            "valid": pd.notna(last_row.get('RM_Entry')),
            "entry": float(last_row.get('RM_Entry', 0)),
            "stop_loss": float(last_row.get('RM_Stop_Loss', 0)),
            "take_profit": float(last_row.get('RM_Take_Profit', 0)),
            "risk_reward": float(last_row.get('RM_Risk_Reward', 0)),
            "position_size": int(last_row.get('RM_Position_Size', 0))
        },
        "events": get_earnings_context(request.symbol, request.analysis_date),
        "market_regime": get_market_regime(provider, request.analysis_date)
    }
    
    # Evaluar decisión
    gemini_client = GeminiClient() if request.use_gemini else None
    
    decision_result = generate_decision(
        analysis_state=analysis_state,
        use_gemini=request.use_gemini,
        gemini_client=gemini_client
    )
    
    return {
        **analysis_state,
        **decision_result
    }

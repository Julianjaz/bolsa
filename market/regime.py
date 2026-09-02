import pandas as pd
from typing import Dict, Any
from market.data import MarketDataProvider, get_market_state_as_of
from market.indicators import add_technical_indicators

def get_market_regime(provider: MarketDataProvider, analysis_date: str, reference_symbol: str = "SPY") -> Dict[str, Any]:
    """
    Determina el régimen del mercado usando un índice de referencia.
    """
    df = get_market_state_as_of(provider, reference_symbol, analysis_date, lookback_days=100)
    
    if df.empty or len(df) < 50:
        return {
            "symbol": reference_symbol,
            "trend": "UNKNOWN",
            "volatility": "UNKNOWN"
        }
        
    df = add_technical_indicators(df)
    last_row = df.iloc[-1]
    
    # 1. Trend Regime
    if last_row['EMA_20_gt_EMA_50']:
        if last_row['Close'] > last_row['EMA_20']:
            trend = "BULLISH"
        else:
            trend = "NEUTRAL_BULLISH"
    else:
        if last_row['Close'] < last_row['EMA_20']:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL_BEARISH"
            
    # 2. Volatility Regime
    atr_pct = last_row['ATR_pct']
    # Umbrales estáticos aproximados para SPY (mejorable con percentiles históricos)
    if atr_pct < 1.0:
        vol = "LOW_VOLATILITY"
    elif atr_pct < 2.0:
        vol = "NORMAL_VOLATILITY"
    else:
        vol = "HIGH_VOLATILITY"
        
    return {
        "symbol": reference_symbol,
        "trend": trend,
        "volatility": vol
    }

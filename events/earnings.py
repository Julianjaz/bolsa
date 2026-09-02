import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def get_earnings_context(symbol: str, analysis_date: str) -> Dict[str, Any]:
    """
    Obtiene el contexto de próximos earnings. 
    Nota: yfinance no siempre provee el historial perfecto point-in-time de CUÁNDO
    se anunció la fecha de earnings. Para el MVP usamos la fecha más cercana disponible.
    """
    analysis_dt = pd.to_datetime(analysis_date)
    
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.get_earnings_dates(limit=20) # Historico y futuro cercano
        
        if calendar is None or calendar.empty:
            return {
                "days_until_earnings": None,
                "earnings_risk_flag": False,
                "next_earnings_date": None
            }
            
        # Limpiar timezone
        if calendar.index.tz is not None:
            calendar.index = calendar.index.tz_localize(None)
            
        # Queremos el próximo earnings *después* de la fecha de análisis
        future_earnings = calendar[calendar.index > analysis_dt].sort_index()
        
        if future_earnings.empty:
            return {
                "days_until_earnings": None,
                "earnings_risk_flag": False,
                "next_earnings_date": None
            }
            
        next_date = future_earnings.index[0]
        days_until = (next_date - analysis_dt).days
        
        # Hard Rule: Bloquear entradas si faltan 5 días o menos
        risk_flag = days_until <= 5
        
        return {
            "days_until_earnings": days_until,
            "earnings_risk_flag": risk_flag,
            "next_earnings_date": next_date.strftime("%Y-%m-%d")
        }
        
    except Exception as e:
        logger.warning(f"No se pudieron obtener earnings para {symbol}: {e}")
        return {
            "days_until_earnings": None,
            "earnings_risk_flag": False,
            "next_earnings_date": None
        }

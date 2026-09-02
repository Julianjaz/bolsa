import pandas as pd
import numpy as np

def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detecta patrones de Swing Trading: Breakout, Pullback, Consolidation.
    Requiere que df ya contenga indicadores y support/resistance.
    """
    df = df.copy()
    
    # 1. Breakout Bullish
    # Cierre cruza la resistencia más cercana por encima, con volumen por encima del promedio
    df['Breakout_Bullish'] = (
        (df['Close'] > df['Nearest_Resistance']) & 
        (df['Close'].shift(1) <= df['Nearest_Resistance'].shift(1)) &
        (df['Volume_Ratio'] >= 1.2)  # Configurable threshold
    )
    
    # 2. Breakout Bearish
    df['Breakout_Bearish'] = (
        (df['Close'] < df['Nearest_Support']) & 
        (df['Close'].shift(1) >= df['Nearest_Support'].shift(1)) &
        (df['Volume_Ratio'] >= 1.2)
    )
    
    # 3. Pullback Bullish
    # Tendencia alcista (EMA20 > EMA50), precio se acerca a EMA20, RSI saludable
    df['Pullback_Bullish'] = (
        (df['EMA_20'] > df['EMA_50']) &
        (df['Close'] > df['EMA_50']) &
        (abs(df['Close'] - df['EMA_20']) / df['EMA_20'] < 0.015) & # Se acerca un 1.5% a la EMA20
        (df['Close'].shift(1) > df['EMA_20'].shift(1)) & # Venía de arriba
        (df['RSI_14'] > 40) & (df['RSI_14'] < 70)
    )
    
    # 4. Pullback Bearish
    df['Pullback_Bearish'] = (
        (df['EMA_20'] < df['EMA_50']) &
        (df['Close'] < df['EMA_50']) &
        (abs(df['Close'] - df['EMA_20']) / df['EMA_20'] < 0.015) &
        (df['Close'].shift(1) < df['EMA_20'].shift(1)) &
        (df['RSI_14'] < 60) & (df['RSI_14'] > 30)
    )
    
    # 5. Consolidation
    # Rango estrecho en las últimas N velas, ATR bajo, EMAs muy juntas
    N_consolidation = 10
    rolling_max = df['High'].rolling(N_consolidation).max()
    rolling_min = df['Low'].rolling(N_consolidation).min()
    range_pct = (rolling_max - rolling_min) / rolling_min
    
    df['Consolidation'] = (
        (range_pct < 0.05) & # Rango de 10 días menor al 5%
        (abs(df['EMA_distance_pct']) < 0.02) # EMAs muy juntas
    )
    
    # Guardamos el setup sugerido
    def get_setup(row):
        if pd.isna(row['Breakout_Bullish']): return "NONE"
        if row['Breakout_Bullish']: return "BREAKOUT_BULLISH"
        if row['Breakout_Bearish']: return "BREAKOUT_BEARISH"
        if row['Pullback_Bullish']: return "PULLBACK_BULLISH"
        if row['Pullback_Bearish']: return "PULLBACK_BEARISH"
        if row['Consolidation']: return "CONSOLIDATION"
        return "NONE"
        
    df['Setup'] = df.apply(get_setup, axis=1)
    
    return df

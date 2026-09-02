import pandas as pd
from typing import Dict, Any

SCORE_CONFIG = {
    "trend_weight": 25,
    "momentum_weight": 20,
    "volume_weight": 15,
    "pattern_weight": 20,
    "structure_weight": 20
}

def calculate_technical_score(row: pd.Series, is_long: bool = True) -> Dict[str, Any]:
    """
    Calcula un score técnico determinístico de 0 a 100.
    """
    trend_score = 0
    momentum_score = 0
    volume_score = 0
    pattern_score = 0
    structure_score = 0
    
    reasons = []

    if is_long:
        # Trend (max 25)
        if row.get('EMA_20_gt_EMA_50', False):
            trend_score += 15
            reasons.append("EMA 20 is above EMA 50")
        if row.get('ADX', 0) > 25 and row.get('+DI', 0) > row.get('-DI', 0):
            trend_score += 10
            reasons.append("Strong ADX trend (ADX > 25 and +DI > -DI)")

        # Momentum (max 20)
        if 40 <= row.get('RSI_14', 0) <= 70:
            momentum_score += 10
            reasons.append("RSI is in healthy bullish zone (40-70)")
        if row.get('MACD_bullish', False):
            momentum_score += 10
            reasons.append("MACD histogram is bullish")

        # Volume (max 15)
        if row.get('Volume_Ratio', 0) >= 1.2:
            volume_score += 15
            reasons.append("High volume (Ratio >= 1.2)")
        elif row.get('Volume_Ratio', 0) >= 1.0:
            volume_score += 5
            reasons.append("Normal volume (Ratio >= 1.0)")

        # Pattern (max 20)
        setup = row.get('Setup', 'NONE')
        if setup in ['BREAKOUT_BULLISH', 'PULLBACK_BULLISH']:
            pattern_score += 20
            reasons.append(f"Bullish pattern detected: {setup}")
        elif setup == 'CONSOLIDATION':
            pattern_score += 10
            reasons.append("Consolidation pattern detected")

        # Structure (max 20)
        # Evaluamos distancia a soporte/resistencia
        dist_res = row.get('Distance_to_Resistance_pct', 0)
        if dist_res > 0.05 or pd.isna(dist_res):
            structure_score += 10
            reasons.append("Plenty of room to nearest resistance (>5%)")
        
        dist_sup = row.get('Distance_to_Support_pct', 0)
        if 0 < dist_sup < 0.05:
            structure_score += 10
            reasons.append("Close to support, good risk/reward (<5%)")
            
    else:
        # Lógica SHORT similar invertida
        if not row.get('EMA_20_gt_EMA_50', True):
            trend_score += 15
            reasons.append("EMA 20 is below EMA 50")
        if row.get('ADX', 0) > 25 and row.get('-DI', 0) > row.get('+DI', 0):
            trend_score += 10
            reasons.append("Strong ADX trend (ADX > 25 and -DI > +DI)")

        if 30 <= row.get('RSI_14', 100) <= 60:
            momentum_score += 10
            reasons.append("RSI is in healthy bearish zone (30-60)")
        if row.get('MACD_bearish', False):
            momentum_score += 10
            reasons.append("MACD histogram is bearish")

        if row.get('Volume_Ratio', 0) >= 1.2:
            volume_score += 15
            reasons.append("High volume (Ratio >= 1.2)")

        setup = row.get('Setup', 'NONE')
        if setup in ['BREAKOUT_BEARISH', 'PULLBACK_BEARISH']:
            pattern_score += 20
            reasons.append(f"Bearish pattern detected: {setup}")

        dist_sup = row.get('Distance_to_Support_pct', 0)
        if dist_sup > 0.05 or pd.isna(dist_sup):
            structure_score += 10
            reasons.append("Plenty of room to nearest support (>5%)")
            
        dist_res = row.get('Distance_to_Resistance_pct', 0)
        if 0 < dist_res < 0.05:
            structure_score += 10
            reasons.append("Close to resistance, good risk/reward (<5%)")

    total_score = trend_score + momentum_score + volume_score + pattern_score + structure_score

    return {
        "trend_score": trend_score,
        "momentum_score": momentum_score,
        "volume_score": volume_score,
        "pattern_score": pattern_score,
        "structure_score": structure_score,
        "total": total_score,
        "reasons": reasons
    }

def append_technical_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def apply_score(row):
        # Determinamos si el sesgo es alcista o bajista basado en Setup o EMAs
        setup = row.get('Setup', 'NONE')
        is_long = True
        if setup in ['BREAKOUT_BEARISH', 'PULLBACK_BEARISH'] or (setup == 'NONE' and not row.get('EMA_20_gt_EMA_50', True)):
            is_long = False
            
        score_data = calculate_technical_score(row, is_long=is_long)
        return pd.Series([score_data['total'], score_data['reasons'], is_long])
        
    df[['Technical_Score', 'Technical_Score_Reasons', 'Score_Is_Long']] = df.apply(apply_score, axis=1)
    return df

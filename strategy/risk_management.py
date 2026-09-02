import pandas as pd
from typing import Dict, Any, Optional
import numpy as np

def calculate_risk_management(row: pd.Series, 
                              is_long: bool = True, 
                              atr_multiplier: float = 2.0, 
                              risk_reward_target: float = 2.0,
                              capital: float = 10000.0,
                              risk_percentage: float = 0.01) -> Dict[str, Any]:
    """
    Calcula los niveles de riesgo y recompensa determinísticos.
    """
    entry = row['Close'] # Asumimos entrada al cierre
    atr = row['ATR_14']
    support = row.get('Nearest_Support', entry - atr)
    resistance = row.get('Nearest_Resistance', entry + atr)
    
    # 1. Determinar Stop Loss
    if is_long:
        atr_stop = entry - (atr * atr_multiplier)
        structural_stop = support - (atr * 0.2) # Un poco por debajo del soporte
        
        # Elegimos el stop más conservador o el que esté más cerca dependiendo de la configuración
        # Por defecto, usamos el atr_stop si el structural está muy lejos
        stop_loss = max(atr_stop, structural_stop) if structural_stop > entry - (atr * 3) else atr_stop
    else:
        atr_stop = entry + (atr * atr_multiplier)
        structural_stop = resistance + (atr * 0.2)
        
        stop_loss = min(atr_stop, structural_stop) if structural_stop < entry + (atr * 3) else atr_stop
        
    # 2. Calcular Riesgo
    risk_per_share = abs(entry - stop_loss)
    
    # Evitar división por cero
    if risk_per_share == 0 or pd.isna(risk_per_share):
        return {"valid": False, "reason": "Risk per share is 0"}

    # 3. Determinar Take Profit
    if is_long:
        take_profit = entry + (risk_per_share * risk_reward_target)
    else:
        take_profit = entry - (risk_per_share * risk_reward_target)
        
    reward_per_share = abs(take_profit - entry)
    calculated_rr = reward_per_share / risk_per_share
    
    # 4. Position Sizing
    risk_amount = capital * risk_percentage
    position_size = int(risk_amount // risk_per_share)
    
    return {
        "valid": True,
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "risk_per_share": round(risk_per_share, 2),
        "reward_per_share": round(reward_per_share, 2),
        "risk_reward": round(calculated_rr, 2),
        "position_size": position_size,
        "risk_amount": round(risk_amount, 2)
    }

def append_risk_management(df: pd.DataFrame, 
                           atr_multiplier: float = 2.0, 
                           risk_reward_target: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    
    def apply_risk(row):
        # Asume que ya existe 'Score_Is_Long' por append_technical_score
        is_long = row.get('Score_Is_Long', True)
        
        risk_data = calculate_risk_management(
            row, 
            is_long=is_long, 
            atr_multiplier=atr_multiplier, 
            risk_reward_target=risk_reward_target
        )
        
        if risk_data["valid"]:
            return pd.Series([
                risk_data["entry"], 
                risk_data["stop_loss"], 
                risk_data["take_profit"], 
                risk_data["risk_reward"],
                risk_data["position_size"]
            ])
        else:
            return pd.Series([np.nan, np.nan, np.nan, np.nan, 0])
            
    df[['RM_Entry', 'RM_Stop_Loss', 'RM_Take_Profit', 'RM_Risk_Reward', 'RM_Position_Size']] = df.apply(apply_risk, axis=1)
    return df

import pandas as pd
import numpy as np

def detect_swing_pivots(df: pd.DataFrame, pivot_left: int = 3, pivot_right: int = 3) -> pd.DataFrame:
    """
    Detecta Swing Highs y Swing Lows.
    IMPORTANTE: Para evitar look-ahead bias, un pivot sólo se marca después 
    de que hayan pasado 'pivot_right' velas.
    """
    df = df.copy()
    
    df['Swing_High'] = False
    df['Swing_Low'] = False
    df['Pivot_High_Value'] = np.nan
    df['Pivot_Low_Value'] = np.nan

    # Necesitamos suficientes velas para calcular
    if len(df) < pivot_left + pivot_right + 1:
        return df

    # Identificamos usando ventanas rodantes
    # Para high: el valor actual debe ser el maximo de la ventana (left + 1 + right)
    # Pero el pivot no se conoce en 'i', se conoce en 'i + right'.
    
    # Creamos columnas con el max/min de la ventana centrada en cada punto
    window = pivot_left + pivot_right + 1
    
    # max/min usando shift
    highs = df['High'].values
    lows = df['Low'].values
    
    for i in range(pivot_left, len(df) - pivot_right):
        is_high = True
        is_low = True
        
        current_high = highs[i]
        current_low = lows[i]
        
        for j in range(i - pivot_left, i + pivot_right + 1):
            if j == i:
                continue
            if highs[j] > current_high:
                is_high = False
            if lows[j] < current_low:
                is_low = False
                
        # Registramos el pivot NO en 'i' (eso seria look-ahead), sino en 'i + pivot_right'
        # que es la vela donde realmente se confirma la existencia del pivot.
        if is_high:
            df.iloc[i + pivot_right, df.columns.get_loc('Swing_High')] = True
            df.iloc[i + pivot_right, df.columns.get_loc('Pivot_High_Value')] = current_high
            
        if is_low:
            df.iloc[i + pivot_right, df.columns.get_loc('Swing_Low')] = True
            df.iloc[i + pivot_right, df.columns.get_loc('Pivot_Low_Value')] = current_low
            
    # Forward fill los últimos soportes/resistencias confirmados
    df['Nearest_Resistance'] = df['Pivot_High_Value'].ffill()
    df['Nearest_Support'] = df['Pivot_Low_Value'].ffill()
    
    # Distancias porcentuales
    df['Distance_to_Resistance_pct'] = (df['Nearest_Resistance'] - df['Close']) / df['Close']
    df['Distance_to_Support_pct'] = (df['Close'] - df['Nearest_Support']) / df['Close']
    
    return df

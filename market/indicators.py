import pandas as pd
import numpy as np

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def calculate_ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.DataFrame:
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    
    atr = tr.rolling(window=window).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/window).mean() / atr))
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()
    
    return pd.DataFrame({
        'ADX': adx,
        '+DI': plus_di,
        '-DI': minus_di
    })

def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = calculate_ema(macd_line, signal)
    macd_histogram = macd_line - macd_signal
    return pd.DataFrame({
        'MACD_Line': macd_line,
        'MACD_Signal': macd_signal,
        'MACD_Histogram': macd_histogram
    })

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()
    return atr

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega todos los indicadores técnicos necesarios al DataFrame OHLCV de forma in-place.
    """
    df = df.copy()
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    # --- TREND ---
    df['EMA_20'] = calculate_ema(close, 20)
    df['EMA_50'] = calculate_ema(close, 50)
    df['SMA_20'] = calculate_sma(close, 20)
    df['SMA_50'] = calculate_sma(close, 50)
    
    adx_df = calculate_adx(high, low, close, 14)
    df['ADX'] = adx_df['ADX']
    df['+DI'] = adx_df['+DI']
    df['-DI'] = adx_df['-DI']
    
    df['EMA_20_gt_EMA_50'] = df['EMA_20'] > df['EMA_50']
    
    # Pendientes (slope)
    df['EMA_20_slope'] = (df['EMA_20'] - df['EMA_20'].shift(3)) / df['EMA_20'].shift(3)
    df['EMA_50_slope'] = (df['EMA_50'] - df['EMA_50'].shift(3)) / df['EMA_50'].shift(3)
    
    # Distancia entre EMAs
    df['EMA_distance_pct'] = (df['EMA_20'] - df['EMA_50']) / df['EMA_50']
    
    # Crossovers
    df['EMA_crossover_bullish'] = (df['EMA_20'] > df['EMA_50']) & (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1))
    df['EMA_crossover_bearish'] = (df['EMA_20'] < df['EMA_50']) & (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1))
    
    # --- MOMENTUM ---
    df['RSI_14'] = calculate_rsi(close, 14)
    
    macd_df = calculate_macd(close, 12, 26, 9)
    df['MACD_Line'] = macd_df['MACD_Line']
    df['MACD_Signal'] = macd_df['MACD_Signal']
    df['MACD_Histogram'] = macd_df['MACD_Histogram']
    
    df['MACD_bullish'] = df['MACD_Histogram'] > 0
    df['MACD_bearish'] = df['MACD_Histogram'] < 0
    
    # --- VOLATILITY ---
    df['ATR_14'] = calculate_atr(high, low, close, 14)
    df['ATR_pct'] = (df['ATR_14'] / close) * 100
    
    # --- VOLUME ---
    df['Volume_SMA_20'] = calculate_sma(volume, 20)
    df['Volume_Ratio'] = volume / df['Volume_SMA_20']
    
    return df

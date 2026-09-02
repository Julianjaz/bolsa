import pandas as pd

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df = pd.read_parquet("datos_finales.parquet")

df["daily_return"] = df["Close"].pct_change()
df["volatility_5d"] = df["Close"].rolling(5).std()
df["ma_5"] = df["Close"].rolling(window=5).mean()
df["ma_10"] = df["Close"].rolling(window=10).mean()
df["rsi_14"] = compute_rsi(df["Close"], window=14)
df["volume_change"] = df["Volume"].pct_change()
df["sentiment_score_ma3"] = df["sentiment_score"].rolling(3).mean()
df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek  # 0=Lunes
df = df.dropna().reset_index(drop=True)

df.to_parquet("datos_finales_con_features.parquet", index=False)

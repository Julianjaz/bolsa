import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Carga del modelo FinBERT
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Función para aplicar FinBERT y extraer el sentimiento y el puntaje
def get_finbert_sentiment(text):
    if isinstance(text, str) and text.strip() != "":
        result = sentiment_pipeline(text[:512])[0]  # truncar a 512 tokens
        return pd.Series([result['label'], result['score']])
    else:
        return pd.Series([None, None])



df = pd.read_parquet("nvidia_news.parquet")

# Aplica la función al DataFrame
df[['sentiment_label', 'sentiment_score']] = df['news'].apply(get_finbert_sentiment)

sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}
df["sentiment_value"] = df["sentiment_label"].map(sentiment_map)

# Guarda el DataFrame con las nuevas columnas
df.to_parquet("nvidia_news_sentiment.parquet", index=False)

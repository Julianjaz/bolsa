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

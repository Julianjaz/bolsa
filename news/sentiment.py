from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
import torch

# Carga del modelo FinBERT
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Pipeline de análisis de sentimiento
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Función para analizar una lista de noticias
def analyze_sentiment(news_list):
    results = []
    for text in news_list:
        if not text.strip():
            results.append({"label": "NEUTRAL", "score": 0})
            continue
        try:
            output = sentiment_pipeline(text[:512])[0]  # truncamos a 512 tokens
            results.append(output)
        except Exception as e:
            print(f"Error analizando texto: {e}")
            results.append({"label": "NEUTRAL", "score": 0})
    return results

# Convertimos resultados a DataFrame
def process_sentiment_results(news_df):
    sentiments = analyze_sentiment(news_df['new'].tolist())
    sentiment_labels = [s["label"] for s in sentiments]
    sentiment_scores = [s["score"] for s in sentiments]

    news_df["sentiment_label"] = sentiment_labels
    news_df["sentiment_score"] = sentiment_scores

    # Mapeo a números para entrenamiento
    label_to_num = {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}
    news_df["sentiment_value"] = news_df["sentiment_label"].map(label_to_num)

    return news_df


# # df_n = pd.read_parquet("news.parquet")  # si ya tienes noticias guardadas
# df_result = process_sentiment_results(df_n)

# # Agrupamos por fecha para tener un único valor por día
# daily_sentiment = df_result.groupby('date')["sentiment_value"].median().reset_index()

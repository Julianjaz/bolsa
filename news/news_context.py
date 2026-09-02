import pandas as pd
from typing import Dict, Any, List

def build_news_context(news_df: pd.DataFrame, analysis_date: str) -> Dict[str, Any]:
    """
    Construye un resumen estructurado del contexto de noticias, 
    asegurando no incluir noticias futuras (leakage).
    
    Args:
        news_df (pd.DataFrame): DataFrame con noticias pre-descargadas y con sentimiento calculado.
                                Debe tener columnas: 'date', 'news', 'sentiment_label', 'sentiment_score'
        analysis_date (str): Fecha de análisis límite ("YYYY-MM-DD")
    """
    if news_df.empty:
        return {
            "total_articles": 0,
            "sentiment_label": "UNKNOWN",
            "sentiment_score": 0.0,
            "news_data_unavailable": True
        }
        
    # Validar que no hay leakage:
    # Solo tomamos noticias con fecha <= analysis_date
    df_filtered = news_df[pd.to_datetime(news_df['date']).dt.strftime('%Y-%m-%d') <= analysis_date]
    
    if df_filtered.empty:
        return {
            "total_articles": 0,
            "sentiment_label": "UNKNOWN",
            "sentiment_score": 0.0,
            "news_data_unavailable": True
        }
        
    # Solo tomamos las noticias recientes (ej. últimos 5 días disponibles en el df filtrado)
    df_filtered = df_filtered.sort_values(by='date', ascending=False)
    recent_news = df_filtered.head(15) # Max 15 artículos recientes
    
    total = len(recent_news)
    positive = len(recent_news[recent_news['sentiment_label'] == 'positive'])
    negative = len(recent_news[recent_news['sentiment_label'] == 'negative'])
    neutral = len(recent_news[recent_news['sentiment_label'] == 'neutral'])
    
    # Calcular score numérico
    sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
    recent_news['num_score'] = recent_news['sentiment_label'].map(sentiment_map)
    avg_score = recent_news['num_score'].mean()
    
    if avg_score > 0.2:
        overall_label = "BULLISH"
    elif avg_score < -0.2:
        overall_label = "BEARISH"
    else:
        overall_label = "NEUTRAL"
        
    headlines = recent_news['news'].tolist()
    
    return {
        "total_articles": total,
        "positive_articles": positive,
        "negative_articles": negative,
        "neutral_articles": neutral,
        "sentiment_score": round(avg_score, 2),
        "sentiment_label": overall_label,
        "headlines": headlines,
        "news_data_unavailable": False
    }

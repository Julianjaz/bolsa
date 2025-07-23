# TODO: en un futuro debe ser un pipeline que llame a news y stock y obtenga los datos, puede tener la opcion de que si no existe los cree o que siempre los cree.

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_process import prepare
from news import scraper, sentiment
from stock import stock

STOCK = "NVIDIA"
DATE_START = "2025-07-01"
DATE_END = "2025-07-21"

def run_pipeline():

    print(f"Obteniendo datos de noticias para {STOCK} desde {DATE_START} hasta {DATE_END}...")

    df = scraper.get_news_by_date_range(STOCK, DATE_START, DATE_END)
    print("df-----: ",df.head())

    print("Aplicando análisis de sentimiento...")
    # Aplica la función al DataFrame
    df[['sentiment_label', 'sentiment_score']] = df['news'].apply(sentiment.get_finbert_sentiment)


    sentiment_map = {
        "positive": 1,
        "neutral": 0,
        "negative": -1
    }
    df["sentiment_value"] = df["sentiment_label"].map(sentiment_map)
    print("df-----: ",df.head())

    print("Obteniendo datos de acciones...")

    df_stock = stock.obtener_datos_accion("NVDA", DATE_START, DATE_END)
    print("df_stock-----: ",df_stock.head())
    
    print("Preparando los datos...")
    # Paso 1: Preparar los datos
    data = prepare.run(df,df_stock)

    # # Paso 2: Feature Engineering
    # feature_engineer.run_feature_engineering()

    # # Paso 3: Dividir los datos
    # split.run_split()

    # # Paso 4: Entrenar el modelo
    # train.run_train()
    return data

if __name__ == "__main__":
    final_data = run_pipeline()
    print(final_data.head())
    # final_data.to_parquet("datos_finales.parquet", index=False)
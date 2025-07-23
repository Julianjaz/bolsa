import pandas as pd

def run(news,stock):
    """
    Prepara el DataFrame final combinando los datos de noticias y bolsa.
    """
    # --- 0. Cargar los DataFrames ---

    df_noticias = news
    df_bolsa = stock

    # --- 1. Prepara el dataframe de noticias ---
    df_noticias['date'] = pd.to_datetime(df_noticias['date']).dt.date  # quitar hora
    df_noticias['sentiment_value'] = df_noticias['sentiment_value'].astype(float)

    # --- 2. Agrupar sentimientos por fecha ---
    df_sentimientos_diarios = df_noticias.groupby('date').agg({
        'sentiment_value': 'mean',  # Promedio diario de sentimiento
        'sentiment_score': 'mean',  # Opcional: score medio también
    }).reset_index()

    # --- 3. Prepara df_bolsa ---
    df_bolsa['Date'] = pd.to_datetime(df_bolsa['Date']).dt.date  # aseguramos que sea del mismo tipo

    # --- 4. Hacer merge ---
    df_final = pd.merge(df_bolsa, df_sentimientos_diarios, how='left', left_on='Date', right_on='date')

    # --- 5. Limpiar ---
    df_final.drop(columns=['date'], inplace=True)

    # --- 6. Guardar el DataFrame final ---
    # df_final.to_parquet("datos_finales.parquet", index=False)

    return df_final
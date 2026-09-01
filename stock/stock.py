import yfinance as yf
import pandas as pd

def obtener_datos_accion(ticker="NVDA", fecha_inicio="2022-01-01", fecha_fin="2024-12-31"):
    # Descargar los datos (sin group_by para evitar complicaciones)
    df = yf.download(ticker, start=fecha_inicio, end=fecha_fin)

    # Aplanar columnas si hay MultiIndex (por seguridad)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Verificar que la columna 'Close' existe
    if 'Close' not in df.columns:
        raise KeyError("La columna 'Close' no está en los datos descargados.")

    # Reiniciar índice (fecha como columna)
    df = df.reset_index()

    # Crear variable target
    df['close_tomorrow'] = df['Close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['Close']).astype(int)

    # Limpiar
    df = df.dropna(subset=['close_tomorrow']).drop(columns=['close_tomorrow'])

    return df

if __name__ == "__main__":

    df_stock = obtener_datos_accion("NVDA", "2025-05-01", "2025-07-21")
    print(df_stock)
    df_stock.to_parquet("datos_stock.parquet")

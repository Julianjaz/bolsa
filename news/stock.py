import yfinance as yf
import pandas as pd

def obtener_datos_accion(ticker="NVDA", fecha_inicio="2020-01-01", fecha_fin=None):
    """
    Descarga datos históricos de una acción y calcula si sube o baja al siguiente día.

    Args:
        ticker (str): Ticker de la acción (por ejemplo, 'NVDA').
        fecha_inicio (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        fecha_fin (str or None): Fecha de fin. Si es None, se usa la fecha actual.

    Returns:
        pd.DataFrame: DataFrame con columnas limpias y columna 'target' (0 = baja, 1 = sube).
    """
    if fecha_fin is None:
        fecha_fin = pd.Timestamp.now().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=fecha_inicio, end=fecha_fin)
    df = df.reset_index()
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Calcular si la acción sube al día siguiente
    df['close_tomorrow'] = df['close'].shift(-1)
    df['target'] = (df['close_tomorrow'] > df['close']).astype(int)

    # Eliminar la última fila que tiene NaN en 'close_tomorrow'
    df = df.dropna(subset=['close_tomorrow']).drop(columns=['close_tomorrow'])

    return df



# df_stock = obtener_datos_accion("TSLA", "2022-01-01")
# print(df_stock[['date', 'close', 'target']].tail())


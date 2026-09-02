import yfinance as yf
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class MarketDataProvider(ABC):
    """
    Interfaz abstracta para proveer datos de mercado.
    """
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> pd.DataFrame:
        """
        Descarga datos OHLCV para un símbolo y un rango de fechas.
        
        Args:
            symbol (str): Ticker del activo (ej. "NVDA").
            start_date (str): Fecha de inicio "YYYY-MM-DD".
            end_date (str): Fecha de fin "YYYY-MM-DD".
            timeframe (str): Intervalo de tiempo (ej. "1D", "1h").
            
        Returns:
            pd.DataFrame: DataFrame con columnas estándar: ['Open', 'High', 'Low', 'Close', 'Volume']
                          y el índice debe ser un datetime, ordenado cronológicamente.
        """
        pass


class YahooFinanceProvider(MarketDataProvider):
    """
    Implementación de MarketDataProvider usando la librería yfinance.
    """
    
    def get_historical_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = "1D") -> pd.DataFrame:
        logger.info(f"Descargando datos para {symbol} desde {start_date} hasta {end_date} (TF: {timeframe})")
        
        # Mapeo simple de timeframes, yfinance usa "1d", "1h", etc.
        yf_interval = timeframe.lower()
        
        try:
            df = yf.download(symbol, start=start_date, end=end_date, interval=yf_interval)
            
            if df.empty:
                logger.warning(f"No se encontraron datos para {symbol} en el rango {start_date} - {end_date}")
                return pd.DataFrame()

            # yfinance a veces devuelve MultiIndex columns si descargas un solo ticker en versiones nuevas
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Asegurarse de que tenemos las columnas necesarias
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Falta la columna requerida {col} en los datos de {symbol}")
                    return pd.DataFrame()
            
            # Limpiar datos nulos y asegurarse del orden
            df = df[required_cols].dropna()
            df = df.sort_index()
            
            # Eliminar timezone details para evitar inconsistencias en el backend si no es necesario
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            return df
            
        except Exception as e:
            logger.error(f"Error descargando datos de Yahoo Finance para {symbol}: {e}")
            return pd.DataFrame()


def get_market_state_as_of(provider: MarketDataProvider, symbol: str, analysis_date: str, lookback_days: int = 365, timeframe: str = "1D") -> pd.DataFrame:
    """
    Obtiene los datos de mercado disponibles ESTRICTAMENTE HASTA el analysis_date.
    Esto previene el look-ahead bias (data leakage) en backtesting.
    
    Args:
        provider (MarketDataProvider): Instancia del proveedor de datos.
        symbol (str): Ticker.
        analysis_date (str): Fecha de análisis (ej. "2026-08-31").
        lookback_days (int): Días de historial a traer antes de la fecha de análisis.
        timeframe (str): Intervalo.
        
    Returns:
        pd.DataFrame: DataFrame con datos de mercado, filtrado.
    """
    # Calculamos start_date como analysis_date menos los días de lookback
    analysis_dt = pd.to_datetime(analysis_date)
    start_dt = analysis_dt - pd.Timedelta(days=lookback_days)
    
    start_date_str = start_dt.strftime("%Y-%m-%d")
    # Pedimos hasta analysis_date + 1 día para asegurar que traemos la vela del analysis_date si existe
    end_dt = analysis_dt + pd.Timedelta(days=1)
    end_date_str = end_dt.strftime("%Y-%m-%d")
    
    df = provider.get_historical_data(symbol, start_date_str, end_date_str, timeframe)
    
    if df.empty:
        return df
        
    # Filtrado estricto: Todo registro cuyo índice sea <= analysis_dt
    # Si la fecha de análisis es un datetime sin hora, asumiremos cierre de día.
    # Por seguridad, usaremos la fecha como límite máximo.
    df_filtered = df[df.index <= analysis_dt]
    
    return df_filtered

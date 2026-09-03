"""tests/conftest.py — Fixtures compartidas para todos los tests."""
import pytest
import pandas as pd
import numpy as np
from market.data import MarketDataProvider


class MockMarketProvider(MarketDataProvider):
    """
    Proveedor de datos sintéticos para tests.
    Genera una serie OHLCV determinista sin llamadas externas.
    """

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        if len(dates) == 0:
            return pd.DataFrame()

        np.random.seed(42)
        n = len(dates)

        close = 100 + np.cumsum(np.random.randn(n) * 1.5)
        close = np.maximum(close, 10)  # evitar precios negativos

        high  = close + np.abs(np.random.randn(n) * 0.5)
        low   = close - np.abs(np.random.randn(n) * 0.5)
        low   = np.maximum(low, 1)
        open_ = close + np.random.randn(n) * 0.3

        volume = np.abs(np.random.randn(n) * 1_000_000 + 5_000_000)

        df = pd.DataFrame(
            {
                "Open":   open_,
                "High":   high,
                "Low":    low,
                "Close":  close,
                "Volume": volume,
            },
            index=dates,
        )
        return df


@pytest.fixture
def mock_provider():
    return MockMarketProvider()


@pytest.fixture
def sample_win_trade():
    return {
        "symbol":          "TEST",
        "analysis_date":   "2023-01-03",
        "entry_date":      "2023-01-03",
        "entry_price":     100.0,
        "decision":        "LONG",
        "stop_loss":       95.0,
        "take_profit":     110.0,
        "setup":           "BREAKOUT_BULLISH",
        "technical_score": 70.0,
        "risk_reward":     2.0,
        "exit_date":       "2023-01-06",
        "exit_price":      110.0,
        "result":          "WIN",
        "r_multiple":      2.0,
        "holding_days":    3,
        "ambiguous_candle": False,
    }


@pytest.fixture
def sample_loss_trade():
    return {
        "symbol":          "TEST",
        "analysis_date":   "2023-01-10",
        "entry_date":      "2023-01-10",
        "entry_price":     100.0,
        "decision":        "LONG",
        "stop_loss":       95.0,
        "take_profit":     110.0,
        "setup":           "PULLBACK_BULLISH",
        "technical_score": 60.0,
        "risk_reward":     2.0,
        "exit_date":       "2023-01-12",
        "exit_price":      95.0,
        "result":          "LOSS",
        "r_multiple":      -1.0,
        "holding_days":    2,
        "ambiguous_candle": False,
    }


@pytest.fixture
def sample_short_win_trade():
    return {
        "symbol":          "TEST",
        "analysis_date":   "2023-02-01",
        "entry_date":      "2023-02-01",
        "entry_price":     100.0,
        "decision":        "SHORT",
        "stop_loss":       105.0,
        "take_profit":     90.0,
        "setup":           "BREAKOUT_BEARISH",
        "technical_score": 65.0,
        "risk_reward":     2.0,
        "exit_date":       "2023-02-05",
        "exit_price":      90.0,
        "result":          "WIN",
        "r_multiple":      2.0,
        "holding_days":    4,
        "ambiguous_candle": False,
    }

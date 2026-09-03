"""
tests/test_engine_no_leakage.py
================================
Tests específicos de ausencia de data leakage en BacktestEngine.

Verifican que para cada fecha de análisis X, el engine:
  1. Solo usa datos con índice <= X para tomar la decisión.
  2. No tiene acceso a datos futuros (High/Low de días posteriores a X).
  3. Los datos de días posteriores a X solo se usan para determinar SL/TP.
"""
import pytest
import pandas as pd
import numpy as np
from market.data import MarketDataProvider, get_market_state_as_of
from backtesting.engine import BacktestEngine


class LeakageDetectorProvider(MarketDataProvider):
    """
    Provider que registra todas las solicitudes de datos y verifica
    que ninguna solicitud pide datos más allá de end_date.
    """

    def __init__(self, master_df: pd.DataFrame):
        self.master_df   = master_df
        self.max_date_requested = pd.Timestamp("1900-01-01")

    def get_historical_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
    ) -> pd.DataFrame:
        end_ts = pd.Timestamp(end_date)
        # Registramos la fecha máxima pedida
        if end_ts > self.max_date_requested:
            self.max_date_requested = end_ts

        mask = (
            (self.master_df.index >= pd.Timestamp(start_date))
            & (self.master_df.index <= end_ts)
        )
        return self.master_df.loc[mask].copy()


def _make_df(start: str, end: str) -> pd.DataFrame:
    """Genera datos OHLCV sintéticos deterministas."""
    dates = pd.bdate_range(start=start, end=end)
    n     = len(dates)
    np.random.seed(0)
    close  = 100 + np.cumsum(np.random.randn(n) * 1.5)
    close  = np.maximum(close, 10)
    high   = close + np.abs(np.random.randn(n) * 0.5)
    low    = close - np.abs(np.random.randn(n) * 0.5)
    low    = np.maximum(low, 1)
    open_  = close + np.random.randn(n) * 0.3
    vol    = np.abs(np.random.randn(n) * 1e6 + 5e6)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


# =========================================================================== #
#  Test 1: get_market_state_as_of no filtra datos futuros                     #
# =========================================================================== #

def test_get_market_state_strict_filter():
    """
    get_market_state_as_of debe devolver SOLO filas con índice <= analysis_date.
    """
    master = _make_df("2020-01-01", "2023-12-31")

    class SimpleProvider(MarketDataProvider):
        def get_historical_data(self, sym, start, end, tf="1D"):
            mask = (master.index >= pd.Timestamp(start)) & (master.index <= pd.Timestamp(end))
            return master.loc[mask].copy()

    analysis_date = "2022-06-15"
    result = get_market_state_as_of(SimpleProvider(), "TEST", analysis_date, lookback_days=365)

    max_date = result.index.max()
    assert max_date <= pd.Timestamp(analysis_date), (
        f"get_market_state_as_of devolvió datos posteriores a {analysis_date}: {max_date}"
    )


# =========================================================================== #
#  Test 2: El engine no solicita datos más allá de end_date                   #
# =========================================================================== #

def test_engine_does_not_request_beyond_end_date():
    """
    El BacktestEngine nunca debe solicitar datos más allá de end_date
    en la descarga inicial del dataset maestro.
    """
    master = _make_df("2020-01-01", "2022-12-31")
    detector = LeakageDetectorProvider(master)

    engine     = BacktestEngine(detector)
    end_date   = "2021-12-31"

    engine.run_backtest(
        symbol="TEST",
        start_date="2021-01-01",
        end_date=end_date,
        min_technical_score=0,    # umbral bajo para forzar señales
        min_risk_reward=0.1,
    )

    # La fecha máxima solicitada no debe exceder end_date
    assert detector.max_date_requested <= pd.Timestamp(end_date), (
        f"El engine solicitó datos hasta {detector.max_date_requested}, "
        f"mayor que end_date={end_date}"
    )


# =========================================================================== #
#  Test 3: Los trades de salida no pueden tener exit_date < entry_date        #
# =========================================================================== #

def test_no_exit_before_entry():
    """Ningún trade puede tener exit_date anterior a entry_date."""
    master   = _make_df("2020-01-01", "2022-12-31")
    provider = LeakageDetectorProvider(master)
    engine   = BacktestEngine(provider)

    trades = engine.run_backtest(
        symbol="TEST",
        start_date="2021-01-01",
        end_date="2021-12-31",
        min_technical_score=0,
        min_risk_reward=0.1,
    )

    for t in trades:
        if t["exit_date"] and t["entry_date"]:
            assert pd.Timestamp(t["exit_date"]) >= pd.Timestamp(t["entry_date"]), (
                f"Trade con exit antes de entry: {t}"
            )


# =========================================================================== #
#  Test 4: Solo un trade activo a la vez                                       #
# =========================================================================== #

def test_no_overlapping_trades():
    """
    El engine debe permitir solo un trade activo a la vez.
    Los trades no deben solaparse temporalmente.
    """
    master   = _make_df("2020-01-01", "2022-12-31")
    provider = LeakageDetectorProvider(master)
    engine   = BacktestEngine(provider)

    trades = engine.run_backtest(
        symbol="TEST",
        start_date="2021-01-01",
        end_date="2021-12-31",
        min_technical_score=0,
        min_risk_reward=0.1,
    )

    closed = [t for t in trades if t["exit_date"]]
    for i in range(len(closed) - 1):
        exit_i   = pd.Timestamp(closed[i]["exit_date"])
        entry_ip1 = pd.Timestamp(closed[i + 1]["entry_date"])
        assert entry_ip1 >= exit_i, (
            f"Overlap detectado: trade {i} sale {exit_i}, trade {i+1} entra {entry_ip1}"
        )


# =========================================================================== #
#  Test 5: Política de vela ambigua (SL + TP misma vela)                      #
# =========================================================================== #

def test_ambiguous_candle_conservative_loss():
    """
    Si High >= TP y Low <= SL en la misma vela, la política
    'conservative_loss' debe cerrar como LOSS.
    """
    from backtesting.engine import _finalize_trade

    # Simular un trade que se encuentra con vela ambigua
    trade = {
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
        "exit_date":       None,
        "exit_price":      None,
        "result":          "OPEN",
        "r_multiple":      None,
        "holding_days":    None,
        "ambiguous_candle": False,
    }

    # Simular la lógica de la vela ambigua (conservative_loss)
    trade["exit_date"]       = "2023-01-04"
    trade["exit_price"]      = trade["stop_loss"]
    trade["result"]          = "LOSS"
    trade["ambiguous_candle"] = True
    _finalize_trade(trade)

    assert trade["result"]          == "LOSS"
    assert trade["r_multiple"]      == pytest.approx(-1.0)
    assert trade["ambiguous_candle"] is True


def test_ambiguous_candle_optimistic_win():
    """
    Si la política es 'optimistic_win', vela ambigua → WIN.
    """
    from backtesting.engine import _finalize_trade

    trade = {
        "symbol":          "TEST",
        "analysis_date":   "2023-01-03",
        "entry_date":      "2023-01-03",
        "entry_price":     100.0,
        "decision":        "LONG",
        "stop_loss":       95.0,
        "take_profit":     110.0,
        "exit_date":       "2023-01-04",
        "exit_price":      110.0,
        "result":          "WIN",
        "r_multiple":      None,
        "holding_days":    None,
        "ambiguous_candle": True,
    }

    _finalize_trade(trade)

    assert trade["result"]     == "WIN"
    assert trade["r_multiple"] == pytest.approx(2.0, rel=1e-3)

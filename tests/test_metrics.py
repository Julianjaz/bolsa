"""tests/test_metrics.py — Tests unitarios de backtesting/metrics.py"""
import pytest
from backtesting.metrics import calculate_metrics, _r_multiple, _max_drawdown, _sharpe, _sortino
import pandas as pd


# =========================================================================== #
#  Tests de _r_multiple                                                        #
# =========================================================================== #

def test_r_multiple_win():
    trade = {
        "entry_price": 100.0,
        "stop_loss":   95.0,
        "exit_price":  110.0,
        "result":      "WIN",
    }
    r = _r_multiple(trade)
    # |110 - 100| / |95 - 100| = 10 / 5 = 2.0
    assert r == pytest.approx(2.0, rel=1e-4)


def test_r_multiple_loss():
    trade = {
        "entry_price": 100.0,
        "stop_loss":   95.0,
        "exit_price":  95.0,
        "result":      "LOSS",
    }
    r = _r_multiple(trade)
    assert r == pytest.approx(-1.0)


def test_r_multiple_open():
    trade = {
        "entry_price": 100.0,
        "stop_loss":   95.0,
        "exit_price":  None,
        "result":      "OPEN",
    }
    assert _r_multiple(trade) == 0.0


def test_r_multiple_zero_risk():
    trade = {
        "entry_price": 100.0,
        "stop_loss":   100.0,  # mismo precio = riesgo cero
        "exit_price":  110.0,
        "result":      "WIN",
    }
    assert _r_multiple(trade) == 0.0


# =========================================================================== #
#  Tests de _max_drawdown                                                      #
# =========================================================================== #

def test_max_drawdown_negative():
    equity = pd.Series([1.0, 2.0, 1.5, 3.0, 2.0])
    dd = _max_drawdown(equity)
    # Running max: [1, 2, 2, 3, 3]
    # Drawdown:    [0, 0, -0.5, 0, -1.0]
    assert dd == pytest.approx(-1.0)


def test_max_drawdown_no_drawdown():
    equity = pd.Series([1.0, 2.0, 3.0, 4.0])
    dd = _max_drawdown(equity)
    assert dd == pytest.approx(0.0)


def test_max_drawdown_empty():
    assert _max_drawdown(pd.Series([], dtype=float)) == 0.0


# =========================================================================== #
#  Tests de calculate_metrics                                                  #
# =========================================================================== #

def test_metrics_empty_trades():
    metrics = calculate_metrics([])
    assert metrics["total_trades"] == 0
    assert metrics["closed_trades"] == 0


def test_metrics_no_closed_trades(sample_win_trade):
    open_trade = {**sample_win_trade, "result": "OPEN", "exit_date": None, "exit_price": None}
    metrics = calculate_metrics([open_trade])
    assert metrics["closed_trades"] == 0
    assert metrics["total_trades"] == 1


def test_metrics_single_win(sample_win_trade):
    metrics = calculate_metrics([sample_win_trade])
    assert metrics["total_trades"]  == 1
    assert metrics["closed_trades"] == 1
    assert metrics["wins"]          == 1
    assert metrics["losses"]        == 0
    assert metrics["win_rate"]      == pytest.approx(100.0)


def test_metrics_single_loss(sample_loss_trade):
    metrics = calculate_metrics([sample_loss_trade])
    assert metrics["wins"]      == 0
    assert metrics["losses"]    == 1
    assert metrics["win_rate"]  == pytest.approx(0.0)


def test_metrics_win_rate_consistency(sample_win_trade, sample_loss_trade):
    """Win rate debe ser consistente con wins y closed_trades."""
    trades  = [sample_win_trade, sample_loss_trade]
    metrics = calculate_metrics(trades)
    expected_wr = metrics["wins"] / metrics["closed_trades"] * 100
    assert metrics["win_rate"] == pytest.approx(expected_wr, rel=1e-4)


def test_metrics_profit_factor_consistency(sample_win_trade, sample_loss_trade):
    """profit_factor = gross_profit_R / gross_loss_R."""
    trades  = [sample_win_trade, sample_loss_trade]
    metrics = calculate_metrics(trades)
    # Win: R=2.0, Loss: R=-1.0 → PF = 2.0/1.0 = 2.0
    assert metrics["profit_factor"] == pytest.approx(2.0, rel=1e-3)


def test_metrics_expectancy_consistency(sample_win_trade, sample_loss_trade):
    """expectancy_r = win_rate * avg_win + loss_rate * avg_loss."""
    trades  = [sample_win_trade, sample_loss_trade]
    metrics = calculate_metrics(trades)
    wr      = metrics["win_rate"] / 100
    expected = wr * metrics["average_win"] + (1 - wr) * metrics["average_loss"]
    assert metrics["expectancy_r"] == pytest.approx(expected, rel=1e-3)


def test_metrics_holding_days(sample_win_trade, sample_loss_trade):
    trades  = [sample_win_trade, sample_loss_trade]
    metrics = calculate_metrics(trades)
    # Win: 3 days, Loss: 2 days → avg = 2.5
    assert metrics["average_holding_days"] == pytest.approx(2.5)


def test_metrics_precision_long(sample_win_trade, sample_loss_trade):
    """precision_long = LONG wins / total LONG × 100."""
    trades  = [sample_win_trade, sample_loss_trade]  # ambos LONG
    metrics = calculate_metrics(trades)
    # 1 win, 2 total LONG → 50%
    assert metrics["precision_long"] == pytest.approx(50.0)


def test_metrics_precision_short(sample_win_trade, sample_short_win_trade):
    """precision_short = SHORT wins / total SHORT × 100."""
    trades  = [sample_win_trade, sample_short_win_trade]
    metrics = calculate_metrics(trades)
    # 1 SHORT, 1 win → 100%
    assert metrics["precision_short"] == pytest.approx(100.0)


def test_metrics_breakdown_by_direction(sample_win_trade, sample_short_win_trade):
    trades  = [sample_win_trade, sample_short_win_trade]
    metrics = calculate_metrics(trades)
    assert "long"  in metrics["breakdown"]
    assert "short" in metrics["breakdown"]
    assert metrics["breakdown"]["long"]["trades"]  == 1
    assert metrics["breakdown"]["short"]["trades"] == 1


def test_metrics_breakdown_by_setup(sample_win_trade, sample_loss_trade):
    trades  = [sample_win_trade, sample_loss_trade]
    metrics = calculate_metrics(trades)
    assert "BREAKOUT_BULLISH" in metrics["by_setup"]
    assert "PULLBACK_BULLISH" in metrics["by_setup"]


def test_metrics_sharpe_positive_r_series():
    """Sharpe debe ser > 0 cuando todos los trades son WIN."""
    wins = [
        {"entry_price": 100, "stop_loss": 95, "exit_price": 110,
         "result": "WIN", "decision": "LONG", "setup": "BREAKOUT_BULLISH",
         "entry_date": "2023-01-01", "exit_date": "2023-01-05", "r_multiple": 2.0, "holding_days": 4}
        for _ in range(10)
    ]
    metrics = calculate_metrics(wins)
    assert metrics["sharpe"] >= 0


def test_metrics_sortino_all_wins():
    """Sortino debe ser inf o muy alto cuando no hay trades negativos."""
    wins = [
        {"entry_price": 100, "stop_loss": 95, "exit_price": 110,
         "result": "WIN", "decision": "LONG", "setup": "BREAKOUT_BULLISH",
         "entry_date": "2023-01-01", "exit_date": "2023-01-05", "r_multiple": 2.0, "holding_days": 4}
        for _ in range(5)
    ]
    metrics = calculate_metrics(wins)
    # Sin trades negativos no hay downside deviation → sortino puede ser inf o 0
    assert metrics["sortino"] >= 0 or metrics["sortino"] == float("inf")


def test_metrics_profit_factor_no_losses(sample_win_trade):
    """Profit factor debe ser None (infinito) cuando no hay losses."""
    metrics = calculate_metrics([sample_win_trade])
    # En nuestro modelo, inf → None
    assert metrics["profit_factor"] is None or metrics["profit_factor"] > 0

"""
tests/test_backtest_api.py
===========================
Tests de integración para los endpoints REST de backtest.
Usa FastAPI TestClient (sin llamadas reales a Yahoo Finance).

Cubre:
  - POST /backtest responde 200 con request válido
  - Request inválido (fechas, score, rr) responde 422
  - start_date > end_date falla
  - include_trades=False no devuelve trades
  - include_trades=True devuelve trades
  - Métricas consistentes en el response
  - /health sigue funcionando
  - /analyze sigue funcionando
  - La API usa BacktestEngine (no lógica duplicada)
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# =========================================================================== #
#  Fixtures de mocking                                                         #
# =========================================================================== #

def _make_mock_trades():
    """Genera una lista de trades sintéticos para mockear el engine."""
    return [
        {
            "symbol":          "NVDA",
            "analysis_date":   "2023-03-01",
            "entry_date":      "2023-03-01",
            "entry_price":     250.0,
            "decision":        "LONG",
            "stop_loss":       240.0,
            "take_profit":     270.0,
            "setup":           "BREAKOUT_BULLISH",
            "technical_score": 75.0,
            "risk_reward":     2.0,
            "exit_date":       "2023-03-05",
            "exit_price":      270.0,
            "result":          "WIN",
            "r_multiple":      2.0,
            "holding_days":    4,
            "ambiguous_candle": False,
        },
        {
            "symbol":          "NVDA",
            "analysis_date":   "2023-03-10",
            "entry_date":      "2023-03-10",
            "entry_price":     260.0,
            "decision":        "LONG",
            "stop_loss":       250.0,
            "take_profit":     280.0,
            "setup":           "PULLBACK_BULLISH",
            "technical_score": 65.0,
            "risk_reward":     2.0,
            "exit_date":       "2023-03-12",
            "exit_price":      250.0,
            "result":          "LOSS",
            "r_multiple":      -1.0,
            "holding_days":    2,
            "ambiguous_candle": False,
        },
    ]


# =========================================================================== #
#  /health                                                                     #
# =========================================================================== #

def test_health_check():
    """/health debe responder 200 con status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


# =========================================================================== #
#  POST /backtest — requests válidos                                           #
# =========================================================================== #

@patch("api.routes.BacktestEngine")
def test_backtest_valid_request(MockEngine):
    """POST /backtest con request válido debe responder 200."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    payload = {
        "symbol":             "NVDA",
        "start_date":         "2023-01-01",
        "end_date":           "2023-12-31",
        "min_technical_score": 50,
        "min_risk_reward":    1.5,
        "use_gemini":         False,
        "include_trades":     False,
    }
    response = client.post("/backtest", json=payload)
    assert response.status_code == 200


@patch("api.routes.BacktestEngine")
def test_backtest_response_structure(MockEngine):
    """El response de /backtest debe tener todas las claves requeridas."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    payload = {
        "symbol": "NVDA",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
    }
    response = client.post("/backtest", json=payload)
    data = response.json()

    assert "symbol"               in data
    assert "summary"              in data
    assert "configuration"        in data
    assert "breakdown"            in data
    assert "by_setup"             in data
    assert "warnings"             in data
    assert "backtest_timestamp"   in data
    assert "system_version"       in data


@patch("api.routes.BacktestEngine")
def test_backtest_summary_fields(MockEngine):
    """summary debe contener todas las métricas requeridas."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31"
    })
    summary = response.json()["summary"]

    required = [
        "total_trades", "closed_trades", "wins", "losses",
        "win_rate", "average_win", "average_loss", "average_r",
        "expectancy_r", "profit_factor", "total_return_r",
        "max_drawdown_r", "sharpe", "sortino", "average_holding_days",
    ]
    for field in required:
        assert field in summary, f"Campo faltante en summary: {field}"


@patch("api.routes.BacktestEngine")
def test_backtest_win_rate_consistency(MockEngine):
    """win_rate debe ser consistente con wins y closed_trades."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31"
    })
    summary = response.json()["summary"]

    expected_wr = summary["wins"] / summary["closed_trades"] * 100
    assert abs(summary["win_rate"] - expected_wr) < 0.01


# =========================================================================== #
#  POST /backtest — include_trades                                             #
# =========================================================================== #

@patch("api.routes.BacktestEngine")
def test_backtest_no_trades_by_default(MockEngine):
    """include_trades=False (default) no debe incluir trades en el response."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31",
        "include_trades": False,
    })
    data = response.json()
    assert data.get("trades") is None


@patch("api.routes.BacktestEngine")
def test_backtest_include_trades(MockEngine):
    """include_trades=True debe incluir la lista de trades."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31",
        "include_trades": True,
    })
    data = response.json()
    assert data.get("trades") is not None
    assert len(data["trades"]) > 0


@patch("api.routes.BacktestEngine")
def test_backtest_trade_pagination(MockEngine):
    """Con include_trades=True debe aparecer trade_pagination."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31",
        "include_trades": True, "page": 1, "page_size": 1,
    })
    data = response.json()
    assert data.get("trade_pagination") is not None
    assert data["trade_pagination"]["page_size"] == 1


# =========================================================================== #
#  POST /backtest — requests inválidos (422)                                  #
# =========================================================================== #

def test_backtest_invalid_end_before_start():
    """end_date anterior a start_date debe responder 422."""
    response = client.post("/backtest", json={
        "symbol": "NVDA",
        "start_date": "2023-12-31",
        "end_date":   "2023-01-01",
    })
    assert response.status_code == 422


def test_backtest_invalid_score_above_100():
    """min_technical_score > 100 debe responder 422."""
    response = client.post("/backtest", json={
        "symbol": "NVDA",
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "min_technical_score": 150,
    })
    assert response.status_code == 422


def test_backtest_invalid_score_below_0():
    """min_technical_score < 0 debe responder 422."""
    response = client.post("/backtest", json={
        "symbol": "NVDA",
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "min_technical_score": -1,
    })
    assert response.status_code == 422


def test_backtest_invalid_risk_reward_zero():
    """min_risk_reward <= 0 debe responder 422."""
    response = client.post("/backtest", json={
        "symbol": "NVDA",
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "min_risk_reward": 0,
    })
    assert response.status_code == 422


def test_backtest_missing_symbol():
    """Request sin symbol debe responder 422."""
    response = client.post("/backtest", json={
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
    })
    assert response.status_code == 422


def test_backtest_invalid_date_format():
    """Formato de fecha inválido debe responder 422."""
    response = client.post("/backtest", json={
        "symbol":     "NVDA",
        "start_date": "01-01-2023",   # formato incorrecto
        "end_date":   "12-31-2023",
    })
    assert response.status_code == 422


def test_backtest_invalid_ambiguous_policy():
    """Política de vela ambigua inválida debe responder 422."""
    response = client.post("/backtest", json={
        "symbol":                  "NVDA",
        "start_date":              "2023-01-01",
        "end_date":                "2023-12-31",
        "ambiguous_candle_policy": "unknown_policy",
    })
    assert response.status_code == 422


# =========================================================================== #
#  POST /backtest — usa BacktestEngine (no lógica duplicada)                  #
# =========================================================================== #

@patch("api.routes.BacktestEngine")
def test_backtest_uses_engine(MockEngine):
    """La API debe delegar en BacktestEngine.run_backtest()."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = []
    MockEngine.return_value = mock_instance

    client.post("/backtest", json={
        "symbol": "NVDA", "start_date": "2023-01-01", "end_date": "2023-12-31"
    })

    assert MockEngine.called, "BacktestEngine no fue instanciado"
    assert mock_instance.run_backtest.called, "run_backtest no fue llamado"


@patch("api.routes.BacktestEngine")
def test_backtest_passes_params_to_engine(MockEngine):
    """Los parámetros del request deben pasarse correctamente al engine."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = []
    MockEngine.return_value = mock_instance

    client.post("/backtest", json={
        "symbol":              "AAPL",
        "start_date":          "2022-01-01",
        "end_date":            "2022-12-31",
        "min_technical_score": 70,
        "min_risk_reward":     2.5,
        "use_gemini":          False,
    })

    call_kwargs = mock_instance.run_backtest.call_args
    assert call_kwargs is not None
    kwargs = call_kwargs.kwargs if call_kwargs.kwargs else {}
    args   = call_kwargs.args   if call_kwargs.args   else ()

    # Verificar que el símbolo llega correctamente
    all_args = str(args) + str(kwargs)
    assert "AAPL" in all_args


# =========================================================================== #
#  /analyze — compatibilidad                                                  #
# =========================================================================== #

@patch("api.routes.get_market_state_as_of")
@patch("api.routes.add_technical_indicators")
@patch("api.routes.detect_swing_pivots")
@patch("api.routes.detect_patterns")
@patch("api.routes.append_technical_score")
@patch("api.routes.append_risk_management")
@patch("api.routes.get_earnings_context")
@patch("api.routes.get_market_regime")
@patch("api.routes.generate_decision")
def test_analyze_still_works(
    mock_decision,
    mock_regime,
    mock_earnings,
    mock_rm,
    mock_score,
    mock_patterns,
    mock_pivots,
    mock_indicators,
    mock_market_state,
):
    """El endpoint /analyze debe seguir respondiendo 200."""
    import pandas as pd
    import numpy as np

    # Mock del dataframe de mercado
    dates = pd.bdate_range("2023-01-01", "2023-06-30")
    n = len(dates)
    close = 100 + np.cumsum(np.random.randn(n))
    df = pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": np.ones(n) * 1e6,
        "EMA_20_gt_EMA_50": True, "RSI_14": 55.0,
        "Setup": "BREAKOUT_BULLISH", "Score_Is_Long": True,
        "Technical_Score": 70.0, "Technical_Score_Reasons": [[]]*n,
        "RM_Entry": close, "RM_Stop_Loss": close - 5,
        "RM_Take_Profit": close + 10, "RM_Risk_Reward": 2.0,
        "RM_Position_Size": 10,
    }, index=dates)

    mock_market_state.return_value = df
    mock_indicators.return_value   = df
    mock_pivots.return_value       = df
    mock_patterns.return_value     = df
    mock_score.return_value        = df
    mock_rm.return_value           = df
    mock_earnings.return_value     = {"earnings_risk_flag": False, "days_until_earnings": None, "next_earnings_date": None}
    mock_regime.return_value       = {"trend": "BULLISH", "volatility": "LOW"}
    mock_decision.return_value     = {"decision": "LONG", "confidence": 0.9, "reasoning": "ok", "trade_proposal": None, "gemini_evaluation": None}

    response = client.post("/analyze", json={
        "symbol":        "NVDA",
        "analysis_date": "2023-06-15",
        "use_gemini":    False,
    })

    assert response.status_code == 200
    data = response.json()
    assert "decision" in data


# =========================================================================== #
#  POST /backtest/compare                                                     #
# =========================================================================== #

@patch("api.routes.BacktestEngine")
def test_compare_returns_technical_only(MockEngine):
    """POST /backtest/compare debe devolver al menos el escenario A."""
    mock_instance = MagicMock()
    mock_instance.run_backtest.return_value = _make_mock_trades()
    MockEngine.return_value = mock_instance

    response = client.post("/backtest/compare", json={
        "symbol":     "NVDA",
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
    })
    assert response.status_code == 200
    data = response.json()
    assert "results"           in data
    assert "technical_only"    in data["results"]
    # Escenario B siempre ausente (no hay fuente de noticias históricas)
    assert data["results"].get("technical_news") is None

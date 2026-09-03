"""
backtesting/metrics.py
======================
Calcula todas las métricas de rendimiento del sistema de trading a partir
de una lista de trades cerrados producidos por BacktestEngine.

NOTA SOBRE WIN RATE / PRECISION:
  Win Rate y precision_long/precision_short miden la precisión de las señales
  del sistema (porcentaje de señales que resultaron en ganancia).
  Esto NO es una garantía de rentabilidad futura.

NOTA SOBRE SHARPE/SORTINO:
  Se calcula sobre la serie temporal de R_Multiple por operación.
  risk_free_rate = 0 (documentado).
  No se anualiza sobre días calendario sino sobre número de operaciones
  (metodología apropiada para sistemas de trading discrecional/swing).

NOTA SOBRE DRAWDOWN:
  Se calcula sobre la curva de equity acumulada en unidades de R,
  no sobre el precio del activo subyacente.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


# --------------------------------------------------------------------------- #
#  Helpers internos                                                            #
# --------------------------------------------------------------------------- #

def _r_multiple(trade: Dict[str, Any]) -> float:
    """
    Devuelve el R-múltiple de una operación cerrada (WIN o LOSS).

    Para WIN: |exit_price - entry_price| / |stop_loss - entry_price|
    Para LOSS: -1.0  (un riesgo exacto, política estándar de trading)
    Para EXPIRED/OPEN: 0.0 (neutro — no se incluye en métricas de performance)
    """
    result = trade.get("result", "OPEN")
    if result not in ("WIN", "LOSS"):
        return 0.0

    entry = trade.get("entry_price", trade.get("entry", 0.0)) or 0.0
    stop  = trade.get("stop_loss", 0.0) or 0.0
    risk  = abs(entry - stop)

    if risk == 0:
        return 0.0

    if result == "WIN":
        exit_p = trade.get("exit_price", entry) or entry
        return abs(exit_p - entry) / risk
    else:  # LOSS
        return -1.0


def _equity_curve(r_series: pd.Series) -> pd.Series:
    """Curva de equity acumulada en unidades de R."""
    return r_series.cumsum()


def _max_drawdown(equity: pd.Series) -> float:
    """
    Máximo drawdown sobre la curva de equity (en R).

    Fórmula:  min(equity - running_max) sobre toda la curva.
    Devuelve un número ≤ 0. Si no hay drawdown devuelve 0.0.
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown    = equity - running_max
    return float(drawdown.min())


def _sharpe(r_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio sobre la serie de R por operación.
    risk_free_rate = 0 por defecto (documentado en docstring del módulo).
    """
    if len(r_series) < 2:
        return 0.0
    excess = r_series - risk_free_rate
    std    = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std)


def _sortino(r_series: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Sortino ratio: penaliza solo la volatilidad negativa.
    risk_free_rate = 0 por defecto.
    """
    if len(r_series) < 2:
        return 0.0
    excess   = r_series - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) < 2:
        # Sin suficientes trades negativos para calcular downside deviation
        return float("inf") if excess.mean() > 0 else 0.0
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std)


def _holding_days(trade: Dict[str, Any]) -> Optional[float]:
    """Días entre entry_date y exit_date. None si no hay exit_date."""
    entry_date = trade.get("entry_date") or trade.get("analysis_date")
    exit_date  = trade.get("exit_date")
    if not entry_date or not exit_date:
        return None
    try:
        delta = pd.to_datetime(exit_date) - pd.to_datetime(entry_date)
        return float(delta.days)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
#  Función principal                                                           #
# --------------------------------------------------------------------------- #

def calculate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcula métricas completas de performance a partir de la lista de trades
    devuelta por BacktestEngine.run_backtest().

    Args:
        trades: Lista de dicts con claves:
            symbol, analysis_date, entry_date, entry_price, decision,
            stop_loss, take_profit, setup, technical_score, risk_reward,
            exit_date, exit_price, result (WIN | LOSS | EXPIRED | OPEN)

    Returns:
        Dict con todas las métricas. Si no hay trades devuelve
        {"total_trades": 0, "closed_trades": 0}.
    """
    if not trades:
        return {"total_trades": 0, "closed_trades": 0}

    df = pd.DataFrame(trades)

    total_trades  = len(df)
    closed_df     = df[df["result"].isin(["WIN", "LOSS"])].copy()
    closed_trades = len(closed_df)

    if closed_trades == 0:
        return {
            "total_trades":  total_trades,
            "closed_trades": 0,
            "wins":          0,
            "losses":        0,
        }

    # ----------------------------------------------------------------------- #
    #  R-Multiple por operación                                                #
    # ----------------------------------------------------------------------- #
    closed_df["R_Multiple"] = closed_df.apply(_r_multiple, axis=1)

    wins_df   = closed_df[closed_df["result"] == "WIN"]
    losses_df = closed_df[closed_df["result"] == "LOSS"]

    wins   = len(wins_df)
    losses = len(losses_df)

    win_rate     = wins / closed_trades
    average_r    = float(closed_df["R_Multiple"].mean())
    average_win  = float(wins_df["R_Multiple"].mean())   if wins   > 0 else 0.0
    average_loss = float(losses_df["R_Multiple"].mean()) if losses > 0 else 0.0

    # Expectancy = E[R] = win_rate * avg_win + loss_rate * avg_loss
    expectancy_r = float(
        win_rate * average_win + (1 - win_rate) * average_loss
    )

    # Profit factor: gross_profit / gross_loss en unidades de R
    gross_profit = wins_df["R_Multiple"].sum()   if wins   > 0 else 0.0
    gross_loss   = abs(losses_df["R_Multiple"].sum()) if losses > 0 else 0.0
    profit_factor = (
        round(float(gross_profit / gross_loss), 4)
        if gross_loss > 0
        else float("inf")
    )

    # ----------------------------------------------------------------------- #
    #  Total Return (suma de R acumulados / closed_trades normalizado)         #
    # ----------------------------------------------------------------------- #
    total_return = float(closed_df["R_Multiple"].sum())

    # ----------------------------------------------------------------------- #
    #  Curva de equity y Drawdown                                              #
    # ----------------------------------------------------------------------- #
    equity = _equity_curve(closed_df["R_Multiple"])
    max_dd = _max_drawdown(equity)

    # ----------------------------------------------------------------------- #
    #  Sharpe y Sortino                                                        #
    # ----------------------------------------------------------------------- #
    sharpe  = round(_sharpe(closed_df["R_Multiple"]),  4)
    sortino = round(_sortino(closed_df["R_Multiple"]), 4)

    # ----------------------------------------------------------------------- #
    #  Holding days                                                            #
    # ----------------------------------------------------------------------- #
    holding_days_list = [
        _holding_days(t) for t in trades
        if t.get("result") in ("WIN", "LOSS")
    ]
    holding_days_clean = [d for d in holding_days_list if d is not None]
    avg_holding_days   = (
        round(float(np.mean(holding_days_clean)), 2)
        if holding_days_clean else None
    )

    # ----------------------------------------------------------------------- #
    #  Desglose por dirección (LONG / SHORT)                                   #
    # ----------------------------------------------------------------------- #
    direction_col = "decision"
    breakdown_direction: Dict[str, Any] = {}
    for direction in ("LONG", "SHORT"):
        sub = closed_df[closed_df.get(direction_col, pd.Series(dtype=str)) == direction] \
              if direction_col in closed_df.columns else pd.DataFrame()
        if sub.empty:
            continue
        sub_wins   = sub[sub["result"] == "WIN"]
        sub_losses = sub[sub["result"] == "LOSS"]
        sub_wr     = len(sub_wins) / len(sub) if len(sub) > 0 else 0.0
        sub_avg_w  = float(sub_wins["R_Multiple"].mean())   if not sub_wins.empty   else 0.0
        sub_avg_l  = float(sub_losses["R_Multiple"].mean()) if not sub_losses.empty else 0.0
        breakdown_direction[direction.lower()] = {
            "trades":       len(sub),
            "wins":         len(sub_wins),
            "losses":       len(sub_losses),
            "win_rate":     round(sub_wr * 100, 2),
            "expectancy_r": round(sub_wr * sub_avg_w + (1 - sub_wr) * sub_avg_l, 4),
        }

    # ----------------------------------------------------------------------- #
    #  Desglose por setup                                                      #
    # ----------------------------------------------------------------------- #
    setup_col = "setup"
    breakdown_setup: Dict[str, Any] = {}
    if setup_col in closed_df.columns:
        for setup_name, group in closed_df.groupby(setup_col):
            if not setup_name or setup_name == "NONE":
                continue
            g_wins   = group[group["result"] == "WIN"]
            g_losses = group[group["result"] == "LOSS"]
            g_wr     = len(g_wins) / len(group)
            g_avg_w  = float(g_wins["R_Multiple"].mean())   if not g_wins.empty   else 0.0
            g_avg_l  = float(g_losses["R_Multiple"].mean()) if not g_losses.empty else 0.0
            breakdown_setup[str(setup_name)] = {
                "trades":       len(group),
                "wins":         len(g_wins),
                "losses":       len(g_losses),
                "win_rate":     round(g_wr * 100, 2),
                "expectancy_r": round(g_wr * g_avg_w + (1 - g_wr) * g_avg_l, 4),
            }

    # ----------------------------------------------------------------------- #
    #  Precision por dirección                                                 #
    # ----------------------------------------------------------------------- #
    long_all   = closed_df[closed_df.get(direction_col, pd.Series(dtype=str)) == "LONG"] \
                 if direction_col in closed_df.columns else pd.DataFrame()
    short_all  = closed_df[closed_df.get(direction_col, pd.Series(dtype=str)) == "SHORT"] \
                 if direction_col in closed_df.columns else pd.DataFrame()

    precision_long  = (
        round(len(long_all[long_all["result"] == "WIN"])  / len(long_all)  * 100, 2)
        if not long_all.empty else None
    )
    precision_short = (
        round(len(short_all[short_all["result"] == "WIN"]) / len(short_all) * 100, 2)
        if not short_all.empty else None
    )

    # ----------------------------------------------------------------------- #
    #  Dict de salida                                                          #
    # ----------------------------------------------------------------------- #
    return {
        # Conteos
        "total_trades":          total_trades,
        "closed_trades":         closed_trades,
        "wins":                  wins,
        "losses":                losses,
        # Win rate y precisión
        "win_rate":              round(win_rate * 100, 2),
        "precision_long":        precision_long,
        "precision_short":       precision_short,
        # R-múltiple
        "average_win":           round(average_win,  4),
        "average_loss":          round(average_loss, 4),
        "average_r":             round(average_r,    4),
        "expectancy_r":          round(expectancy_r, 4),
        "profit_factor":         profit_factor if profit_factor != float("inf") else None,
        # Retorno y drawdown (en unidades de R)
        "total_return_r":        round(total_return, 4),
        "max_drawdown_r":        round(max_dd, 4),
        # Ratios de riesgo
        "sharpe":                sharpe,
        "sortino":               sortino,
        # Tiempo
        "average_holding_days":  avg_holding_days,
        # Desglose
        "breakdown":             breakdown_direction,
        "by_setup":              breakdown_setup,
    }

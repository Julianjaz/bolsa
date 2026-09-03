"""
backtesting/engine.py
=====================
Motor de backtesting principal. Es la FUENTE ÚNICA DE VERDAD para
la simulación histórica de operaciones de Swing Trading.

ANTI-LEAKAGE GARANTIZADO
--------------------------
Para cada fecha de análisis X:
  - Solo se usan datos con índice <= X (via get_market_state_as_of).
  - El LocalProvider filtra master_df.index <= end internamente.
  - Los datos de días posteriores a X se usan ÚNICAMENTE para determinar
    si SL o TP fueron alcanzados (resultado de la operación), nunca
    para tomar la decisión de entrar.

POLÍTICA SL/TP AMBIGUO (misma vela)
--------------------------------------
Si en una misma vela futura:
    High >= take_profit  AND  Low <= stop_loss

No es posible determinar cuál ocurrió primero con datos OHLC diarios.
Política por defecto: "conservative_loss" (LOSS).

Rationale: Asumir la peor situación es la única opción honesta para evitar
sobreestimar el rendimiento del sistema. La alternativa ("optimistic_win")
sobrestimaría el sistema.

Configurable a través del parámetro ambiguous_candle_policy:
    - "conservative_loss" (default): LOSS
    - "optimistic_win":              WIN  (no recomendado para validación)
    - "skip":                        EXPIRED (se cierra como neutro)

ADVERTENCIA SOBRE EARNINGS
----------------------------
get_earnings_context() usa yfinance, que puede exponer fechas de earnings
que no eran conocidas públicamente en la fecha de análisis histórica.
Para backtesting esto puede introducir leakage mínimo en el filtro de earnings.
Por eso el engine NO llama get_earnings_context() en el backtesting —
el filtro de earnings solo aplica en el endpoint /analyze (modo real-time).
"""

import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from market.data import MarketDataProvider, get_market_state_as_of
from market.indicators import add_technical_indicators
from market.support_resistance import detect_swing_pivots
from market.patterns import detect_patterns
from strategy.technical_score import append_technical_score
from strategy.risk_management import append_risk_management

logger = logging.getLogger(__name__)

SYSTEM_VERSION = "1.0.0"


class BacktestEngine:
    """
    Motor de backtesting que simula operaciones de Swing Trading sobre
    datos históricos sin data leakage.
    """

    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        min_technical_score: float = 50.0,
        min_risk_reward: float = 1.5,
        use_gemini: bool = False,
        ambiguous_candle_policy: str = "conservative_loss",
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta el backtest completo para un símbolo en un rango de fechas.

        Args:
            symbol:                  Ticker del activo (ej. "NVDA").
            start_date:              Fecha de inicio "YYYY-MM-DD".
            end_date:                Fecha de fin "YYYY-MM-DD".
            min_technical_score:     Puntaje técnico mínimo para entrar (0-100).
            min_risk_reward:         Ratio riesgo/recompensa mínimo.
            use_gemini:              Si True, usa Gemini como filtro adicional.
                                     NOTA: para backtests largos esto puede ser
                                     muy lento y costoso. Se recomienda False.
            ambiguous_candle_policy: Política para velas donde High>=TP y Low<=SL.
                                     Opciones: "conservative_loss" | "optimistic_win" | "skip"

        Returns:
            Lista de trades. Cada trade es un dict con las claves:
                symbol, analysis_date, entry_date, entry_price, decision,
                stop_loss, take_profit, setup, technical_score, risk_reward,
                exit_date, exit_price, result, r_multiple, holding_days
        """
        logger.info(
            f"[BacktestEngine] Iniciando backtest {symbol} "
            f"{start_date} → {end_date} | "
            f"score>={min_technical_score} rr>={min_risk_reward} gemini={use_gemini}"
        )

        # ------------------------------------------------------------------- #
        #  Descarga única del dataset maestro                                  #
        # ------------------------------------------------------------------- #
        # Solo descargamos hasta end_date para no tener datos futuros en memoria
        master_df = self.provider.get_historical_data(symbol, "2000-01-01", end_date)
        if master_df.empty:
            logger.error("[BacktestEngine] No hay datos históricos disponibles.")
            return []

        # ------------------------------------------------------------------- #
        #  LocalProvider: proveedor que sirve slices del master_df             #
        #  Garantía: nunca filtra datos posteriores a end_date.                #
        # ------------------------------------------------------------------- #
        class _LocalProvider(MarketDataProvider):
            def get_historical_data(
                self_lp,
                sym: str,
                start: str,
                end: str,
                tf: str = "1D",
            ) -> pd.DataFrame:
                mask = (
                    (master_df.index >= pd.to_datetime(start))
                    & (master_df.index <= pd.to_datetime(end))
                )
                return master_df.loc[mask].copy()

        local_provider = _LocalProvider()

        # ------------------------------------------------------------------- #
        #  Días de trading dentro del rango pedido                             #
        # ------------------------------------------------------------------- #
        mask_range = (
            (master_df.index >= pd.to_datetime(start_date))
            & (master_df.index <= pd.to_datetime(end_date))
        )
        trading_days = master_df.loc[mask_range].index

        # ------------------------------------------------------------------- #
        #  Configuración Gemini (solo si use_gemini=True)                      #
        # ------------------------------------------------------------------- #
        gemini_client = None
        if use_gemini:
            try:
                from llm.gemini_client import GeminiClient
                gemini_client = GeminiClient()
                if gemini_client.client is None:
                    logger.warning(
                        "[BacktestEngine] use_gemini=True pero GEMINI_API_KEY no configurada. "
                        "Gemini desactivado."
                    )
                    gemini_client = None
            except Exception as e:
                logger.warning(f"[BacktestEngine] No se pudo inicializar GeminiClient: {e}")
                gemini_client = None

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None

        for current_date in trading_days:
            date_str = current_date.strftime("%Y-%m-%d")

            # --------------------------------------------------------------- #
            #  1. ACTUALIZAR TRADE ACTIVO                                      #
            # --------------------------------------------------------------- #
            if active_trade is not None:
                candle = master_df.loc[current_date]
                high   = float(candle["High"])
                low    = float(candle["Low"])

                tp_hit = False
                sl_hit = False

                if active_trade["decision"] == "LONG":
                    tp_hit = high >= active_trade["take_profit"]
                    sl_hit = low  <= active_trade["stop_loss"]
                else:  # SHORT
                    tp_hit = low  <= active_trade["take_profit"]
                    sl_hit = high >= active_trade["stop_loss"]

                # --- Política SL/TP ambiguo ---
                if tp_hit and sl_hit:
                    # No podemos saber cuál ocurrió primero con datos OHLC
                    policy = ambiguous_candle_policy
                    if policy == "conservative_loss":
                        result     = "LOSS"
                        exit_price = active_trade["stop_loss"]
                    elif policy == "optimistic_win":
                        result     = "WIN"
                        exit_price = active_trade["take_profit"]
                    else:  # "skip"
                        result     = "EXPIRED"
                        exit_price = float(candle["Close"])
                    active_trade["exit_date"]  = date_str
                    active_trade["exit_price"] = exit_price
                    active_trade["result"]     = result
                    active_trade["ambiguous_candle"] = True
                    _finalize_trade(active_trade)
                    trades.append(active_trade)
                    active_trade = None
                    continue

                elif sl_hit:
                    active_trade["exit_date"]  = date_str
                    active_trade["exit_price"] = active_trade["stop_loss"]
                    active_trade["result"]     = "LOSS"
                    _finalize_trade(active_trade)
                    trades.append(active_trade)
                    active_trade = None
                    continue

                elif tp_hit:
                    active_trade["exit_date"]  = date_str
                    active_trade["exit_price"] = active_trade["take_profit"]
                    active_trade["result"]     = "WIN"
                    _finalize_trade(active_trade)
                    trades.append(active_trade)
                    active_trade = None
                    continue

            # --------------------------------------------------------------- #
            #  2. BUSCAR NUEVA ENTRADA (solo si no hay trade activo)           #
            # --------------------------------------------------------------- #
            if active_trade is not None:
                continue

            # ANTI-LEAKAGE: obtener solo datos hasta current_date inclusive
            df_slice = get_market_state_as_of(
                local_provider, symbol, date_str, lookback_days=365
            )

            if len(df_slice) < 60:
                continue  # No hay suficientes datos para calcular indicadores

            # Calcular todos los indicadores sobre el slice (sin datos futuros)
            df_slice = add_technical_indicators(df_slice)
            df_slice = detect_swing_pivots(df_slice)
            df_slice = detect_patterns(df_slice)
            df_slice = append_technical_score(df_slice)
            df_slice = append_risk_management(
                df_slice, risk_reward_target=min_risk_reward
            )

            last_row = df_slice.iloc[-1]

            setup = last_row.get("Setup", "NONE")
            score = float(last_row.get("Technical_Score", 0))
            rr    = float(last_row.get("RM_Risk_Reward", 0))

            # Hard rules (espejado de decision_engine para backtesting)
            if (
                setup != "NONE"
                and score >= min_technical_score
                and rr >= min_risk_reward
                and pd.notna(last_row.get("RM_Entry"))
            ):
                is_long   = bool(last_row["Score_Is_Long"])
                direction = "LONG" if is_long else "SHORT"

                # Filtro Gemini opcional
                if use_gemini and gemini_client:
                    try:
                        from strategy.decision_engine import generate_decision
                        analysis_state = {
                            "ticker": symbol,
                            "analysis_date": date_str,
                            "technical": {
                                "close": float(last_row["Close"]),
                                "EMA_20_gt_EMA_50": bool(last_row.get("EMA_20_gt_EMA_50", False)),
                                "RSI": float(last_row.get("RSI_14", 0)),
                                "Setup": setup,
                                "Score_Is_Long": is_long,
                            },
                            "technical_score": score,
                            "risk_management": {
                                "valid": True,
                                "entry": float(last_row.get("RM_Entry", 0)),
                                "stop_loss": float(last_row.get("RM_Stop_Loss", 0)),
                                "take_profit": float(last_row.get("RM_Take_Profit", 0)),
                                "risk_reward": rr,
                                "position_size": int(last_row.get("RM_Position_Size", 0)),
                            },
                            "events": {"earnings_risk_flag": False},
                            "market_regime": {},
                        }
                        gemini_result = generate_decision(
                            analysis_state=analysis_state,
                            min_technical_score=min_technical_score,
                            min_risk_reward=min_risk_reward,
                            use_gemini=True,
                            gemini_client=gemini_client,
                        )
                        if gemini_result.get("decision") == "HOLD":
                            continue  # Gemini vetó la entrada
                    except Exception as e:
                        logger.warning(f"[BacktestEngine] Error Gemini en {date_str}: {e}")

                active_trade = {
                    "symbol":          symbol,
                    "analysis_date":   date_str,
                    "entry_date":      date_str,
                    "entry_price":     float(last_row["RM_Entry"]),
                    "decision":        direction,
                    "stop_loss":       float(last_row["RM_Stop_Loss"]),
                    "take_profit":     float(last_row["RM_Take_Profit"]),
                    "setup":           setup,
                    "technical_score": round(score, 2),
                    "risk_reward":     round(rr, 2),
                    "exit_date":       None,
                    "exit_price":      None,
                    "result":          "OPEN",
                    "r_multiple":      None,
                    "holding_days":    None,
                    "ambiguous_candle": False,
                }

        # ------------------------------------------------------------------- #
        #  Cerrar trade pendiente al final del periodo                         #
        # ------------------------------------------------------------------- #
        if active_trade is not None:
            active_trade["result"] = "EXPIRED"
            _finalize_trade(active_trade)
            trades.append(active_trade)

        logger.info(
            f"[BacktestEngine] Backtest completo: {len(trades)} trades generados."
        )
        return trades


# --------------------------------------------------------------------------- #
#  Helper: finalizar trade (calcular r_multiple y holding_days)               #
# --------------------------------------------------------------------------- #

def _finalize_trade(trade: Dict[str, Any]) -> None:
    """Calcula r_multiple y holding_days en el trade dict (in-place)."""
    entry  = trade.get("entry_price") or 0.0
    stop   = trade.get("stop_loss")   or 0.0
    exit_p = trade.get("exit_price")  or entry
    result = trade.get("result", "OPEN")

    risk = abs(entry - stop)
    if risk > 0 and result in ("WIN", "LOSS"):
        if result == "WIN":
            trade["r_multiple"] = round(abs(exit_p - entry) / risk, 4)
        else:
            trade["r_multiple"] = -1.0
    else:
        trade["r_multiple"] = 0.0

    # holding_days
    entry_date = trade.get("entry_date") or trade.get("analysis_date")
    exit_date  = trade.get("exit_date")
    if entry_date and exit_date:
        try:
            delta = pd.to_datetime(exit_date) - pd.to_datetime(entry_date)
            trade["holding_days"] = int(delta.days)
        except Exception:
            trade["holding_days"] = None

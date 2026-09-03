"""
api/routes.py
=============
Endpoints REST del sistema de Swing Trading.

Endpoints disponibles:
    GET  /health           → Health check
    POST /analyze          → Análisis de un símbolo en una fecha dada
    POST /backtest         → Backtesting histórico con métricas completas
    POST /backtest/compare → Comparación de 3 escenarios (Technical Only /
                             Technical+News / Technical+News+Gemini)

ARQUITECTURA:
    API → Pydantic request → BacktestEngine.run_backtest() → calculate_metrics() → Response
    La API solo orquesta. No duplica lógica de trading ni de métricas.

NOTA SOBRE LEAKAGE:
    El endpoint /backtest NO llama get_earnings_context() por fecha histórica
    para evitar leakage potencial de datos de earnings (yfinance no es
    point-in-time para fechas pasadas). Se documenta como warning en el response.
"""

import datetime
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List

from market.data import YahooFinanceProvider, get_market_state_as_of
from market.indicators import add_technical_indicators
from market.support_resistance import detect_swing_pivots
from market.patterns import detect_patterns
from strategy.technical_score import append_technical_score
from strategy.risk_management import append_risk_management
from events.earnings import get_earnings_context
from market.regime import get_market_regime
from llm.gemini_client import GeminiClient
from strategy.decision_engine import generate_decision
from backtesting.engine import BacktestEngine, SYSTEM_VERSION
from backtesting.metrics import calculate_metrics

router = APIRouter()


# =========================================================================== #
#  MODELOS DE REQUEST                                                          #
# =========================================================================== #

class AnalyzeRequest(BaseModel):
    """Request para el endpoint /analyze."""
    symbol: str = Field(..., description="Ticker del activo (ej. NVDA)")
    timeframe: str = Field("1D", description="Intervalo temporal")
    analysis_date: Optional[str] = Field(
        None, description="Fecha de análisis YYYY-MM-DD. Por defecto: hoy."
    )
    use_gemini: bool = Field(False, description="Activar filtro Gemini")

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()


class BacktestRequest(BaseModel):
    """
    Request para el endpoint POST /backtest.

    Example:
        {
            "symbol": "NVDA",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "min_technical_score": 50,
            "min_risk_reward": 1.5,
            "use_gemini": false,
            "include_trades": false,
            "page": 1,
            "page_size": 100
        }
    """
    symbol: str = Field(
        ...,
        description="Ticker del activo, ej. NVDA",
        examples=["NVDA"],
    )
    start_date: str = Field(
        ...,
        description="Fecha de inicio del backtest YYYY-MM-DD",
        examples=["2023-01-01"],
    )
    end_date: str = Field(
        ...,
        description="Fecha de fin del backtest YYYY-MM-DD",
        examples=["2023-12-31"],
    )
    min_technical_score: float = Field(
        50.0,
        ge=0,
        le=100,
        description="Score técnico mínimo para activar una señal (0-100)",
    )
    min_risk_reward: float = Field(
        1.5,
        gt=0,
        description="Ratio riesgo/recompensa mínimo (> 0)",
    )
    use_gemini: bool = Field(
        False,
        description="Activar filtro Gemini. ADVERTENCIA: muy lento para rangos largos.",
    )
    ambiguous_candle_policy: str = Field(
        "conservative_loss",
        description=(
            "Política para velas donde SL y TP se alcanzan en la misma vela. "
            "Opciones: 'conservative_loss' (default) | 'optimistic_win' | 'skip'"
        ),
    )
    include_trades: bool = Field(
        False,
        description="Si True, incluye la lista de trades en el response.",
    )
    page: int = Field(1, ge=1, description="Página de trades (si include_trades=True)")
    page_size: int = Field(
        100, ge=1, le=1000, description="Tamaño de página de trades"
    )

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("ambiguous_candle_policy")
    @classmethod
    def validate_policy(cls, v: str) -> str:
        allowed = {"conservative_loss", "optimistic_win", "skip"}
        if v not in allowed:
            raise ValueError(f"ambiguous_candle_policy debe ser uno de {allowed}")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestRequest":
        try:
            start = datetime.date.fromisoformat(self.start_date)
            end   = datetime.date.fromisoformat(self.end_date)
        except ValueError as e:
            raise ValueError(f"Formato de fecha inválido: {e}")
        if end <= start:
            raise ValueError("end_date debe ser posterior a start_date")
        return self


class BacktestCompareRequest(BaseModel):
    """Request para el endpoint POST /backtest/compare."""
    symbol: str = Field(..., description="Ticker del activo, ej. NVDA")
    start_date: str = Field(..., description="Fecha de inicio YYYY-MM-DD")
    end_date: str = Field(..., description="Fecha de fin YYYY-MM-DD")
    min_technical_score: float = Field(50.0, ge=0, le=100)
    min_risk_reward: float = Field(1.5, gt=0)

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def validate_dates(self) -> "BacktestCompareRequest":
        try:
            start = datetime.date.fromisoformat(self.start_date)
            end   = datetime.date.fromisoformat(self.end_date)
        except ValueError as e:
            raise ValueError(f"Formato de fecha inválido: {e}")
        if end <= start:
            raise ValueError("end_date debe ser posterior a start_date")
        return self


# =========================================================================== #
#  MODELOS DE RESPONSE                                                         #
# =========================================================================== #

class TradeRecord(BaseModel):
    """Registro de una operación individual del backtest."""
    symbol: str
    analysis_date: str
    entry_date: str
    direction: str
    setup_type: str
    entry: float
    stop_loss: float
    take_profit: float
    exit_date: Optional[str]
    exit_price: Optional[float]
    result: str
    r_multiple: Optional[float]
    holding_days: Optional[int]
    technical_score: Optional[float]
    risk_reward: Optional[float]
    ambiguous_candle: bool = False


class BacktestSummary(BaseModel):
    total_trades: int
    closed_trades: int
    wins: int
    losses: int
    win_rate: float
    precision_long: Optional[float]
    precision_short: Optional[float]
    average_win: float
    average_loss: float
    average_r: float
    expectancy_r: float
    profit_factor: Optional[float]
    total_return_r: float
    max_drawdown_r: float
    sharpe: float
    sortino: float
    average_holding_days: Optional[float]


class DirectionBreakdown(BaseModel):
    trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_r: float


class SetupBreakdown(BaseModel):
    trades: int
    wins: int
    losses: int
    win_rate: float
    expectancy_r: float


class BacktestConfiguration(BaseModel):
    min_technical_score: float
    min_risk_reward: float
    use_gemini: bool
    ambiguous_candle_policy: str


class BacktestResponse(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    configuration: BacktestConfiguration
    backtest_timestamp: str
    system_version: str
    summary: BacktestSummary
    breakdown: Dict[str, DirectionBreakdown]
    by_setup: Dict[str, SetupBreakdown]
    trades: Optional[List[TradeRecord]]
    trade_pagination: Optional[Dict[str, int]]
    warnings: List[str]


class CompareResult(BaseModel):
    technical_only: Optional[Dict[str, Any]]
    technical_news: Optional[Dict[str, Any]]
    technical_news_gemini: Optional[Dict[str, Any]]


class BacktestCompareResponse(BaseModel):
    symbol: str
    period: Dict[str, str]
    results: CompareResult
    warnings: List[str]


# =========================================================================== #
#  HELPERS                                                                     #
# =========================================================================== #

def _trade_to_record(trade: Dict[str, Any]) -> TradeRecord:
    """Convierte un trade dict del engine al modelo de response."""
    return TradeRecord(
        symbol=trade.get("symbol", ""),
        analysis_date=trade.get("analysis_date", ""),
        entry_date=trade.get("entry_date", ""),
        direction=trade.get("decision", ""),
        setup_type=trade.get("setup", "NONE"),
        entry=trade.get("entry_price", 0.0),
        stop_loss=trade.get("stop_loss", 0.0),
        take_profit=trade.get("take_profit", 0.0),
        exit_date=trade.get("exit_date"),
        exit_price=trade.get("exit_price"),
        result=trade.get("result", "OPEN"),
        r_multiple=trade.get("r_multiple"),
        holding_days=trade.get("holding_days"),
        technical_score=trade.get("technical_score"),
        risk_reward=trade.get("risk_reward"),
        ambiguous_candle=trade.get("ambiguous_candle", False),
    )


def _build_backtest_response(
    request: BacktestRequest,
    trades: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    warnings: List[str],
) -> BacktestResponse:
    """Construye el BacktestResponse completo desde trades + metrics."""
    summary = BacktestSummary(
        total_trades=metrics.get("total_trades", 0),
        closed_trades=metrics.get("closed_trades", 0),
        wins=metrics.get("wins", 0),
        losses=metrics.get("losses", 0),
        win_rate=metrics.get("win_rate", 0.0),
        precision_long=metrics.get("precision_long"),
        precision_short=metrics.get("precision_short"),
        average_win=metrics.get("average_win", 0.0),
        average_loss=metrics.get("average_loss", 0.0),
        average_r=metrics.get("average_r", 0.0),
        expectancy_r=metrics.get("expectancy_r", 0.0),
        profit_factor=metrics.get("profit_factor"),
        total_return_r=metrics.get("total_return_r", 0.0),
        max_drawdown_r=metrics.get("max_drawdown_r", 0.0),
        sharpe=metrics.get("sharpe", 0.0),
        sortino=metrics.get("sortino", 0.0),
        average_holding_days=metrics.get("average_holding_days"),
    )

    breakdown_raw = metrics.get("breakdown", {})
    breakdown = {
        k: DirectionBreakdown(**v) for k, v in breakdown_raw.items()
    }

    by_setup_raw = metrics.get("by_setup", {})
    by_setup = {
        k: SetupBreakdown(**v) for k, v in by_setup_raw.items()
    }

    trade_records = None
    trade_pagination = None
    if request.include_trades:
        total_t = len(trades)
        start_i = (request.page - 1) * request.page_size
        end_i   = start_i + request.page_size
        page_trades = trades[start_i:end_i]
        trade_records = [_trade_to_record(t) for t in page_trades]
        trade_pagination = {
            "total":     total_t,
            "page":      request.page,
            "page_size": request.page_size,
            "pages":     max(1, (total_t + request.page_size - 1) // request.page_size),
        }

    return BacktestResponse(
        symbol=request.symbol,
        start_date=request.start_date,
        end_date=request.end_date,
        configuration=BacktestConfiguration(
            min_technical_score=request.min_technical_score,
            min_risk_reward=request.min_risk_reward,
            use_gemini=request.use_gemini,
            ambiguous_candle_policy=request.ambiguous_candle_policy,
        ),
        backtest_timestamp=datetime.datetime.utcnow().isoformat(),
        system_version=SYSTEM_VERSION,
        summary=summary,
        breakdown=breakdown,
        by_setup=by_setup,
        trades=trade_records,
        trade_pagination=trade_pagination,
        warnings=warnings,
    )


# =========================================================================== #
#  ENDPOINTS                                                                   #
# =========================================================================== #

@router.get("/health", summary="Health Check")
def health_check():
    """Verifica que la API esté activa."""
    return {"status": "ok", "service": "Swing Trading API", "version": SYSTEM_VERSION}


@router.post("/analyze", summary="Análisis de símbolo en fecha dada")
def analyze_symbol(request: AnalyzeRequest):
    """
    Ejecuta el pipeline completo de análisis para un símbolo en una fecha.
    Incluye indicadores técnicos, earnings risk, market regime y decisión opcional con Gemini.
    """
    provider = YahooFinanceProvider()

    if not request.analysis_date:
        request.analysis_date = datetime.datetime.now().strftime("%Y-%m-%d")

    df = get_market_state_as_of(provider, request.symbol, request.analysis_date, lookback_days=365)

    if df.empty or len(df) < 60:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INSUFFICIENT_DATA",
                "message": f"Datos insuficientes para {request.symbol}",
                "details": {"rows": len(df)},
            },
        )

    df = add_technical_indicators(df)
    df = detect_swing_pivots(df)
    df = detect_patterns(df)
    df = append_technical_score(df)
    df = append_risk_management(df)

    last_row = df.iloc[-1]

    analysis_state = {
        "ticker":         request.symbol,
        "analysis_date":  request.analysis_date,
        "timeframe":      request.timeframe,
        "technical": {
            "close":          float(last_row["Close"]),
            "EMA_20_gt_EMA_50": bool(last_row.get("EMA_20_gt_EMA_50", False)),
            "RSI":            float(last_row.get("RSI_14", 0)),
            "Setup":          str(last_row.get("Setup", "NONE")),
            "Score_Is_Long":  bool(last_row.get("Score_Is_Long", True)),
        },
        "technical_score": float(last_row.get("Technical_Score", 0)),
        "risk_management": {
            "valid":         pd.notna(last_row.get("RM_Entry")),
            "entry":         float(last_row.get("RM_Entry", 0)),
            "stop_loss":     float(last_row.get("RM_Stop_Loss", 0)),
            "take_profit":   float(last_row.get("RM_Take_Profit", 0)),
            "risk_reward":   float(last_row.get("RM_Risk_Reward", 0)),
            "position_size": int(last_row.get("RM_Position_Size", 0)),
        },
        "events":       get_earnings_context(request.symbol, request.analysis_date),
        "market_regime": get_market_regime(provider, request.analysis_date),
    }

    gemini_client = GeminiClient() if request.use_gemini else None

    decision_result = generate_decision(
        analysis_state=analysis_state,
        use_gemini=request.use_gemini,
        gemini_client=gemini_client,
    )

    return {**analysis_state, **decision_result}


@router.post(
    "/backtest",
    response_model=BacktestResponse,
    summary="Backtesting histórico",
    response_model_exclude_none=True,
)
def run_backtest(request: BacktestRequest):
    """
    Ejecuta un backtest completo usando BacktestEngine y devuelve métricas
    completas: win rate, expectancy, Sharpe, Sortino, max drawdown, desglose
    por dirección y setup.

    **Anti-leakage**: Para cada fecha X solo se usan datos conocidos hasta X.
    Los datos posteriores a X se usan solo para determinar el resultado (SL/TP).

    **Política SL/TP ambiguo**: Si en la misma vela High>=TP y Low<=SL,
    la política por defecto es `conservative_loss` (LOSS).

    **Earnings en backtesting**: El engine no filtra por earnings en modo backtest
    porque yfinance no es point-in-time para fechas históricas. Se agrega warning.
    """
    warnings: List[str] = [
        "Earnings data from yfinance may not be perfectly point-in-time for historical dates. "
        "Earnings filter is disabled in backtest mode to prevent forward-looking bias.",
        f"SL/TP ambiguous candle policy: '{request.ambiguous_candle_policy}'. "
        "Same-candle SL+TP hits are resolved by this policy.",
    ]

    if request.use_gemini:
        warnings.append(
            "use_gemini=true: Gemini adds a per-trade API call. "
            "Large date ranges may be very slow and expensive. "
            "GEMINI_API_KEY must be set in environment."
        )

    try:
        provider = YahooFinanceProvider()
        engine   = BacktestEngine(provider)

        trades = engine.run_backtest(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            min_technical_score=request.min_technical_score,
            min_risk_reward=request.min_risk_reward,
            use_gemini=request.use_gemini,
            ambiguous_candle_policy=request.ambiguous_candle_policy,
        )

        if not trades:
            warnings.append(
                f"No se encontraron datos históricos para {request.symbol} "
                f"en el rango {request.start_date} – {request.end_date}."
            )
            # Devolvemos un response vacío pero válido
            empty_metrics: Dict[str, Any] = {
                "total_trades": 0, "closed_trades": 0, "wins": 0, "losses": 0,
                "win_rate": 0.0, "precision_long": None, "precision_short": None,
                "average_win": 0.0, "average_loss": 0.0, "average_r": 0.0,
                "expectancy_r": 0.0, "profit_factor": None,
                "total_return_r": 0.0, "max_drawdown_r": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "average_holding_days": None,
                "breakdown": {}, "by_setup": {},
            }
            return _build_backtest_response(request, [], empty_metrics, warnings)

        metrics = calculate_metrics(trades)
        return _build_backtest_response(request, trades, metrics, warnings)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "BACKTEST_FAILED",
                "message": str(e),
                "details": {},
            },
        )


@router.post(
    "/backtest/compare",
    response_model=BacktestCompareResponse,
    summary="Comparación de estrategias (Technical / News / Gemini)",
    response_model_exclude_none=True,
)
def compare_strategies(request: BacktestCompareRequest):
    """
    Ejecuta tres escenarios de backtesting para comparar el valor agregado
    de noticias y Gemini sobre la señal técnica pura.

    **Escenario A**: Technical Only  
    **Escenario B**: Technical + News — ADVERTENCIA: no hay fuente de noticias
    históricas point-in-time disponible. Se devuelve warning y el escenario
    se omite para evitar leakage.  
    **Escenario C**: Technical + News + Gemini — se ejecuta solo si
    GEMINI_API_KEY está configurada.
    """
    warnings: List[str] = [
        "Earnings filter is disabled in backtest mode (yfinance earnings not point-in-time).",
    ]

    try:
        provider = YahooFinanceProvider()
        engine   = BacktestEngine(provider)

        # Escenario A: Technical Only
        trades_a = engine.run_backtest(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            min_technical_score=request.min_technical_score,
            min_risk_reward=request.min_risk_reward,
            use_gemini=False,
        )
        metrics_a = calculate_metrics(trades_a)

        # Escenario B: Technical + News
        # No disponible sin fuente de noticias históricas point-in-time
        warnings.append(
            "Scenario B (Technical + News) skipped: no historical point-in-time "
            "news source available. Using current news for past dates would "
            "introduce look-ahead bias."
        )
        metrics_b = None

        # Escenario C: Technical + News + Gemini
        metrics_c = None
        try:
            gc_check = GeminiClient()
            if gc_check.client is None:
                warnings.append(
                    "Scenario C (Technical + News + Gemini) skipped: "
                    "GEMINI_API_KEY not configured."
                )
            else:
                warnings.append(
                    "Scenario C (Technical + Gemini) running. "
                    "This may take a long time for large date ranges."
                )
                trades_c = engine.run_backtest(
                    symbol=request.symbol,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    min_technical_score=request.min_technical_score,
                    min_risk_reward=request.min_risk_reward,
                    use_gemini=True,
                )
                metrics_c = calculate_metrics(trades_c)
        except Exception as e:
            warnings.append(f"Scenario C failed: {e}")

        return BacktestCompareResponse(
            symbol=request.symbol,
            period={"start": request.start_date, "end": request.end_date},
            results=CompareResult(
                technical_only=metrics_a,
                technical_news=metrics_b,
                technical_news_gemini=metrics_c,
            ),
            warnings=warnings,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error":   "COMPARE_FAILED",
                "message": str(e),
                "details": {},
            },
        )

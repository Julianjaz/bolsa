"""
backtesting/run.py
==================
CLI para ejecutar el BacktestEngine directamente desde la terminal,
usando exactamente el mismo motor que el endpoint POST /backtest.

Uso:
    python -m backtesting.run --symbol NVDA --start 2023-01-01 --end 2023-12-31

Opciones:
    --symbol             Ticker del activo (requerido)
    --start              Fecha de inicio YYYY-MM-DD (requerido)
    --end                Fecha de fin YYYY-MM-DD (requerido)
    --min-score          Score técnico mínimo (default: 50)
    --min-rr             Risk/Reward mínimo (default: 1.5)
    --use-gemini         Activar filtro Gemini (flag, default: False)
    --policy             Política SL/TP ambiguo (default: conservative_loss)
    --format             Formato de salida: json | table (default: table)
    --include-trades     Incluir lista de trades en salida (flag)

Ejemplo completo:
    python -m backtesting.run \\
        --symbol NVDA \\
        --start 2023-01-01 \\
        --end 2023-12-31 \\
        --min-score 50 \\
        --min-rr 1.5 \\
        --format json \\
        --include-trades
"""

import argparse
import json
import sys
import logging

# Importaciones del proyecto (mismo motor que la API)
from market.data import YahooFinanceProvider
from backtesting.engine import BacktestEngine, SYSTEM_VERSION
from backtesting.metrics import calculate_metrics

logging.basicConfig(
    level=logging.WARNING,  # CLI: solo errores, para no saturar la terminal
    format="%(levelname)s: %(message)s",
)


def _print_table(symbol: str, start: str, end: str, metrics: dict, trades: list, include_trades: bool):
    """Imprime los resultados en formato tabla legible."""
    divider = "=" * 60

    print(f"\n{divider}")
    print(f"  BACKTEST RESULTS — {symbol}  ({start} → {end})")
    print(f"  System version: {SYSTEM_VERSION}")
    print(divider)

    summary_fields = [
        ("Total trades",          metrics.get("total_trades",         "—")),
        ("Closed trades",         metrics.get("closed_trades",        "—")),
        ("Wins",                  metrics.get("wins",                 "—")),
        ("Losses",                metrics.get("losses",               "—")),
        ("Win rate",              f"{metrics.get('win_rate', 0):.2f}%"),
        ("Precision LONG",        f"{metrics.get('precision_long', '—')}%"
                                  if metrics.get("precision_long") is not None else "—"),
        ("Precision SHORT",       f"{metrics.get('precision_short', '—')}%"
                                  if metrics.get("precision_short") is not None else "—"),
        ("Average win (R)",       f"{metrics.get('average_win', 0):.4f}"),
        ("Average loss (R)",      f"{metrics.get('average_loss', 0):.4f}"),
        ("Average R",             f"{metrics.get('average_r', 0):.4f}"),
        ("Expectancy (R)",        f"{metrics.get('expectancy_r', 0):.4f}"),
        ("Profit factor",         f"{metrics.get('profit_factor', '—')}"),
        ("Total return (R)",      f"{metrics.get('total_return_r', 0):.4f}"),
        ("Max drawdown (R)",      f"{metrics.get('max_drawdown_r', 0):.4f}"),
        ("Sharpe",                f"{metrics.get('sharpe', 0):.4f}"),
        ("Sortino",               f"{metrics.get('sortino', 0):.4f}"),
        ("Avg holding days",      f"{metrics.get('average_holding_days', '—')}"),
    ]

    for label, value in summary_fields:
        print(f"  {label:<25} {value}")

    breakdown = metrics.get("breakdown", {})
    if breakdown:
        print(f"\n  {'─'*55}")
        print("  BREAKDOWN BY DIRECTION")
        print(f"  {'─'*55}")
        for direction, data in breakdown.items():
            print(
                f"  {direction.upper():<8} trades={data['trades']} "
                f"wins={data['wins']} losses={data['losses']} "
                f"wr={data['win_rate']:.1f}% "
                f"exp={data['expectancy_r']:.4f}R"
            )

    by_setup = metrics.get("by_setup", {})
    if by_setup:
        print(f"\n  {'─'*55}")
        print("  BREAKDOWN BY SETUP")
        print(f"  {'─'*55}")
        for setup, data in by_setup.items():
            print(
                f"  {setup:<25} trades={data['trades']} "
                f"wr={data['win_rate']:.1f}% "
                f"exp={data['expectancy_r']:.4f}R"
            )

    if include_trades and trades:
        print(f"\n  {'─'*55}")
        print("  TRADES")
        print(f"  {'─'*55}")
        header = f"  {'Entry':<12} {'Exit':<12} {'Dir':<6} {'Setup':<22} {'R':<8} {'Result'}"
        print(header)
        for t in trades:
            r = t.get("r_multiple")
            r_str = f"{r:.2f}" if r is not None else "—"
            print(
                f"  {t.get('entry_date','—'):<12} "
                f"{t.get('exit_date','—'):<12} "
                f"{t.get('decision','—'):<6} "
                f"{t.get('setup','NONE'):<22} "
                f"{r_str:<8} "
                f"{t.get('result','—')}"
            )

    print(f"\n{divider}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m backtesting.run",
        description="Swing Trading Backtester — CLI",
    )
    parser.add_argument("--symbol",         required=True,  help="Ticker del activo (ej. NVDA)")
    parser.add_argument("--start",          required=True,  help="Fecha de inicio YYYY-MM-DD")
    parser.add_argument("--end",            required=True,  help="Fecha de fin YYYY-MM-DD")
    parser.add_argument("--min-score",      type=float, default=50.0, help="Score técnico mínimo (default: 50)")
    parser.add_argument("--min-rr",         type=float, default=1.5,  help="Risk/Reward mínimo (default: 1.5)")
    parser.add_argument("--use-gemini",     action="store_true", default=False, help="Activar filtro Gemini")
    parser.add_argument("--policy",         default="conservative_loss",
                        choices=["conservative_loss", "optimistic_win", "skip"],
                        help="Política SL/TP ambiguo (default: conservative_loss)")
    parser.add_argument("--format",         default="table", choices=["json", "table"],
                        help="Formato de salida: json | table (default: table)")
    parser.add_argument("--include-trades", action="store_true", default=False,
                        help="Incluir lista completa de trades")

    args = parser.parse_args()

    symbol = args.symbol.strip().upper()

    print(f"\n[BacktestCLI] Iniciando backtest para {symbol} {args.start} → {args.end} …", file=sys.stderr)

    provider = YahooFinanceProvider()
    engine   = BacktestEngine(provider)

    try:
        trades = engine.run_backtest(
            symbol=symbol,
            start_date=args.start,
            end_date=args.end,
            min_technical_score=args.min_score,
            min_risk_reward=args.min_rr,
            use_gemini=args.use_gemini,
            ambiguous_candle_policy=args.policy,
        )
    except Exception as e:
        print(f"[ERROR] Backtest falló: {e}", file=sys.stderr)
        sys.exit(1)

    metrics = calculate_metrics(trades)

    if args.format == "json":
        output = {
            "symbol":      symbol,
            "start_date":  args.start,
            "end_date":    args.end,
            "configuration": {
                "min_technical_score": args.min_score,
                "min_risk_reward":     args.min_rr,
                "use_gemini":          args.use_gemini,
                "ambiguous_candle_policy": args.policy,
            },
            "system_version": SYSTEM_VERSION,
            "summary":     metrics,
        }
        if args.include_trades:
            output["trades"] = trades
        print(json.dumps(output, indent=2, default=str))
    else:
        _print_table(
            symbol=symbol,
            start=args.start,
            end=args.end,
            metrics=metrics,
            trades=trades,
            include_trades=args.include_trades,
        )


if __name__ == "__main__":
    main()

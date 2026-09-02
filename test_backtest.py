import logging
from market.data import YahooFinanceProvider
from backtesting.engine import BacktestEngine
from backtesting.metrics import calculate_metrics

logging.basicConfig(level=logging.INFO)

def test_backtest():
    provider = YahooFinanceProvider()
    engine = BacktestEngine(provider)
    
    symbol = "NVDA"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    trades = engine.run_backtest(symbol, start_date, end_date)
    metrics = calculate_metrics(trades)
    
    print(f"\n--- Backtest Results for {symbol} ({start_date} to {end_date}) ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    print(f"\nFirst 3 trades:")
    for t in trades[:3]:
        print(f"Entry: {t['entry_date']} ({t['decision']} @ {t['entry_price']}) -> Exit: {t['exit_date']} @ {t['exit_price']} [{t['result']}] (Setup: {t['setup']}, Score: {t['technical_score']})")

if __name__ == "__main__":
    test_backtest()

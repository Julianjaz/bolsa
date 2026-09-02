import logging
from market.data import YahooFinanceProvider, get_market_state_as_of

logging.basicConfig(level=logging.INFO)

def test_market_data():
    provider = YahooFinanceProvider()
    
    symbol = "NVDA"
    analysis_date = "2024-01-10"  # Random past date
    
    print(f"Testing fetch for {symbol} as of {analysis_date}...")
    df = get_market_state_as_of(provider, symbol, analysis_date, lookback_days=10)
    
    print(df)
    
    if not df.empty:
        last_date = df.index[-1].strftime("%Y-%m-%d")
        print(f"Last date in dataframe: {last_date}")
        assert last_date <= analysis_date, "Data leakage detected: returned future data!"
        print("Success! No future data returned.")
    else:
        print("No data returned.")

if __name__ == "__main__":
    test_market_data()

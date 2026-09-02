import pandas as pd
from typing import List, Dict, Any

def calculate_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"number_of_trades": 0}
        
    df = pd.DataFrame(trades)
    
    # Filtramos trades expirados o abiertos sin resultado
    closed_trades = df[df['result'].isin(['WIN', 'LOSS'])].copy()
    
    if closed_trades.empty:
        return {"number_of_trades": len(df), "closed_trades": 0}
        
    closed_trades['profit'] = closed_trades.apply(
        lambda row: abs(row['exit_price'] - row['entry_price']) if row['result'] == 'WIN' 
        else -abs(row['exit_price'] - row['entry_price']), 
        axis=1
    )
    
    # Simplified calculation since position sizing is dynamic. We use standard risk_reward multiple tracking.
    closed_trades['R_Multiple'] = closed_trades.apply(
        lambda row: abs(row['exit_price'] - row['entry_price']) / abs(row['stop_loss'] - row['entry_price']) if row['result'] == 'WIN' else -1,
        axis=1
    )
    
    wins = closed_trades[closed_trades['result'] == 'WIN']
    losses = closed_trades[closed_trades['result'] == 'LOSS']
    
    win_rate = len(wins) / len(closed_trades)
    average_R = closed_trades['R_Multiple'].mean()
    expectancy = (win_rate * wins['R_Multiple'].mean()) - ((1 - win_rate) * 1) if not wins.empty else -1
    
    gross_profit = wins['profit'].sum() if not wins.empty else 0
    gross_loss = abs(losses['profit'].sum()) if not losses.empty else 0
    
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    metrics = {
        "number_of_trades": len(df),
        "closed_trades": len(closed_trades),
        "win_rate": round(win_rate * 100, 2),
        "wins": len(wins),
        "losses": len(losses),
        "average_R": round(average_R, 2),
        "expectancy_R": round(expectancy, 2),
        "profit_factor": round(profit_factor, 2)
    }
    
    return metrics

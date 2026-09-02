import pandas as pd
import logging
from typing import List, Dict, Any
from market.data import MarketDataProvider, get_market_state_as_of
from market.indicators import add_technical_indicators
from market.support_resistance import detect_swing_pivots
from market.patterns import detect_patterns
from strategy.technical_score import append_technical_score
from strategy.risk_management import append_risk_management

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, provider: MarketDataProvider):
        self.provider = provider

    def run_backtest(self, 
                     symbol: str, 
                     start_date: str, 
                     end_date: str, 
                     min_technical_score: int = 50,
                     min_risk_reward: float = 1.5) -> List[Dict[str, Any]]:
        
        logger.info(f"Running backtest for {symbol} from {start_date} to {end_date}")
        
        # Para evitar llamadas a la API de yfinance por cada día, descargamos el dataset maestro
        # y creamos un proveedor local (MockProvider) que actúe sobre este dataset.
        master_df = self.provider.get_historical_data(symbol, "2000-01-01", end_date)
        if master_df.empty:
            logger.error("No historical data available for backtest.")
            return []
            
        class LocalProvider(MarketDataProvider):
            def get_historical_data(self, sym: str, start: str, end: str, tf: str = "1D"):
                mask = (master_df.index >= pd.to_datetime(start)) & (master_df.index <= pd.to_datetime(end))
                return master_df.loc[mask].copy()

        local_provider = LocalProvider()
        
        # Fechas a evaluar (solo los días de trading dentro del rango)
        mask_range = (master_df.index >= pd.to_datetime(start_date)) & (master_df.index <= pd.to_datetime(end_date))
        trading_days = master_df.loc[mask_range].index
        
        trades = []
        active_trade = None
        
        for current_date in trading_days:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # --- 1. ACTUALIZAR TRADE ACTIVO ---
            if active_trade is not None:
                current_candle = master_df.loc[current_date]
                high = current_candle['High']
                low = current_candle['Low']
                
                # Check Stop Loss / Take Profit
                if active_trade['decision'] == 'LONG':
                    if low <= active_trade['stop_loss']:
                        active_trade['exit_date'] = date_str
                        active_trade['exit_price'] = active_trade['stop_loss']
                        active_trade['result'] = 'LOSS'
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    elif high >= active_trade['take_profit']:
                        active_trade['exit_date'] = date_str
                        active_trade['exit_price'] = active_trade['take_profit']
                        active_trade['result'] = 'WIN'
                        trades.append(active_trade)
                        active_trade = None
                        continue
                else: # SHORT
                    if high >= active_trade['stop_loss']:
                        active_trade['exit_date'] = date_str
                        active_trade['exit_price'] = active_trade['stop_loss']
                        active_trade['result'] = 'LOSS'
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    elif low <= active_trade['take_profit']:
                        active_trade['exit_date'] = date_str
                        active_trade['exit_price'] = active_trade['take_profit']
                        active_trade['result'] = 'WIN'
                        trades.append(active_trade)
                        active_trade = None
                        continue
            
            # Solo buscamos nuevas entradas si no hay un trade activo
            if active_trade is None:
                # --- 2. PUNTO EN EL TIEMPO: OBTENER DATOS SIN LEAKAGE ---
                # Usamos 365 días de lookback para tener suficientes datos para EMAs largas y Pivots
                df_slice = get_market_state_as_of(local_provider, symbol, date_str, lookback_days=365)
                
                if len(df_slice) < 60:
                    continue # No hay datos suficientes
                    
                # --- 3. CALCULAR INDICADORES SOBRE EL SLICE ---
                df_slice = add_technical_indicators(df_slice)
                df_slice = detect_swing_pivots(df_slice)
                df_slice = detect_patterns(df_slice)
                df_slice = append_technical_score(df_slice)
                df_slice = append_risk_management(df_slice, risk_reward_target=min_risk_reward)
                
                # Evaluamos SOLO la última vela
                last_row = df_slice.iloc[-1]
                
                setup = last_row.get('Setup', 'NONE')
                score = last_row.get('Technical_Score', 0)
                rr = last_row.get('RM_Risk_Reward', 0)
                
                # Hard rules for technicals
                if setup != 'NONE' and score >= min_technical_score and rr >= min_risk_reward:
                    is_long = last_row['Score_Is_Long']
                    
                    active_trade = {
                        'symbol': symbol,
                        'analysis_date': date_str,
                        'entry_date': date_str, # Simplificación: Entramos al cierre del mismo día
                        'entry_price': last_row['RM_Entry'],
                        'decision': 'LONG' if is_long else 'SHORT',
                        'stop_loss': last_row['RM_Stop_Loss'],
                        'take_profit': last_row['RM_Take_Profit'],
                        'setup': setup,
                        'technical_score': score,
                        'risk_reward': rr,
                        'exit_date': None,
                        'exit_price': None,
                        'result': 'OPEN'
                    }

        # Cerrar el trade pendiente al final si existe
        if active_trade is not None:
            active_trade['result'] = 'EXPIRED'
            trades.append(active_trade)
            
        return trades


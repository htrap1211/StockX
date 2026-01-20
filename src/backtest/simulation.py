import pandas as pd
import numpy as np
import vectorbt as vbt
from typing import Dict, Any

class BacktestEngine:
    def __init__(self):
        pass

    def run_simulation(self, price_data: pd.DataFrame, preds_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs a vectorbt simulation based on model predictions.
        
        Args:
            price_data: DataFrame with 'close', 'open' etc. Index=Date.
            preds_df: DataFrame with 'prob' (0-1). Index=Date.
            
        Strategy:
            - Long Entry: Prob > 0.6
            - Long Exit: Prob < 0.4 due to signal decay OR Stop Loss hit.
        """
        # Align indices
        common_idx = price_data.index.intersection(preds_df.index)
        if len(common_idx) == 0:
            return {"error": "No overlapping dates between Price and Preds"}
            
        price = price_data.loc[common_idx, 'close']
        probs = preds_df.loc[common_idx, 'prob']
        
        # 1. Generate Entry/Exit Signals
        entries = probs > 0.6
        exits = probs < 0.4
        
        # 2. Run Portfolio Simulation
        # Settings:
        # - Fees: 0.1% per trade (conservative est for slippage+comm)
        # - Sl_stop: 5%
        # - Freq: '1D'
        portfolio = vbt.Portfolio.from_signals(
            close=price,
            entries=entries,
            exits=exits,
            sl_stop=0.05,
            # tp_stop = 0.10, # Optional: Take profit at 10%?
            fees=0.001,
            freq='1D',
            init_cash=100_000,
            size=0.1,         # 10% of equity per trade
            size_type='percent' 
        )
        
        # 3. Extract Metrics
        stats = portfolio.stats()
        
        # Return summary dictionary
        results = {
            "total_return": stats.get('Total Return [%]', 0),
            "benchmark_return": stats.get('Benchmark Return [%]', 0),
            "max_drawdown": stats.get('Max Drawdown [%]', 0),
            "sharpe_ratio": stats.get('Sharpe Ratio', 0),
            "win_rate": stats.get('Win Rate [%]', 0),
            "total_trades": stats.get('Total Trades', 0),
            "portfolio_object": portfolio # Return actual object for plotting if needed
        }
        
        return results

    def generate_report(self, results: Dict[str, Any]):
        print("\n--- Backtest Report ---")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Benchmark Return: {results['benchmark_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print("-----------------------")

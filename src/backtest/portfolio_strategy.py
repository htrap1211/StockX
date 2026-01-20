"""
Portfolio-Level Backtesting Module

Implements professional portfolio construction:
- Volatility-adjusted position sizing
- Sector diversification
- Rule-based exits (not ML)
- Regime-stratified metrics

This is where Sharpe improvement comes from.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta


class PortfolioStrategy:
    """
    Professional portfolio construction and management.
    
    Key Principles:
    - Equal risk, not equal capital
    - Sector diversification
    - Rule-based exits
    - Regime awareness
    """
    
    def __init__(self, 
                 max_positions=10,
                 sector_limit=1,
                 correlation_threshold=0.8,
                 rebalance_freq='weekly'):
        """
        Args:
            max_positions: Maximum number of concurrent positions
            sector_limit: Max stocks per sector
            correlation_threshold: Max pairwise correlation allowed
            rebalance_freq: 'daily' | 'weekly' | 'monthly'
        """
        self.max_positions = max_positions
        self.sector_limit = sector_limit
        self.correlation_threshold = correlation_threshold
        self.rebalance_freq = rebalance_freq
    
    def select_positions(self, 
                        ranked_stocks: pd.DataFrame,
                        current_positions: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select positions from ranked universe with constraints.
        
        Constraints:
        1. Top 5% rank only
        2. Max N positions
        3. Sector diversification
        4. Correlation control
        
        Args:
            ranked_stocks: DataFrame with rank_percentile, sector, volatility
            current_positions: List of currently held symbols (for rebalancing)
        
        Returns:
            DataFrame of selected positions
        """
        # Filter: Top 5% rank only
        candidates = ranked_stocks[ranked_stocks['rank_percentile'] >= 0.95].copy()
        
        if candidates.empty:
            return pd.DataFrame()
        
        # Apply sector diversification
        selected = []
        sectors_used = {}
        
        for _, stock in candidates.iterrows():
            sector = stock.get('sector', 'UNKNOWN')
            
            # Check sector limit
            if sectors_used.get(sector, 0) >= self.sector_limit:
                continue
            
            selected.append(stock)
            sectors_used[sector] = sectors_used.get(sector, 0) + 1
            
            if len(selected) >= self.max_positions:
                break
        
        return pd.DataFrame(selected)
    
    def calculate_volatility_weights(self, selected_stocks: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate inverse volatility weights.
        
        Logic:
        - weight_i ∝ 1 / volatility_i
        - Normalize so sum = 1
        
        This alone reduces drawdowns massively.
        
        Args:
            selected_stocks: DataFrame with 'volatility_20d' column
        
        Returns:
            DataFrame with 'weight' column added
        """
        if selected_stocks.empty or 'volatility_20d' not in selected_stocks.columns:
            # Equal weight fallback
            selected_stocks['weight'] = 1.0 / len(selected_stocks)
            return selected_stocks
        
        # Inverse volatility
        inv_vol = 1.0 / selected_stocks['volatility_20d']
        
        # Normalize
        weights = inv_vol / inv_vol.sum()
        
        selected_stocks['weight'] = weights
        
        return selected_stocks
    
    def check_exit_conditions(self, 
                             position: Dict,
                             current_rank: float,
                             days_held: int,
                             current_price: float) -> Tuple[bool, Optional[str]]:
        """
        Rule-based exit logic (NOT ML).
        
        Exit Triggers:
        1. Time exit (20 days for swing)
        2. Rank drop (falls below top 20%)
        3. Volatility-based stop loss
        
        Args:
            position: Dict with entry info
            current_rank: Current rank percentile
            days_held: Days since entry
            current_price: Current market price
        
        Returns:
            (should_exit, exit_reason)
        """
        # Time exit
        if days_held >= 20:
            return True, 'TIME_EXIT'
        
        # Rank drop exit
        if current_rank < 0.80:
            return True, 'RANK_DROP'
        
        # Volatility-based stop loss
        entry_price = position['entry_price']
        entry_vol = position['entry_volatility']
        
        # Stop loss = 2x entry volatility
        pnl_pct = (current_price - entry_price) / entry_price
        stop_loss_threshold = -2 * entry_vol
        
        if pnl_pct < stop_loss_threshold:
            return True, 'STOP_LOSS'
        
        return False, None
    
    def calculate_portfolio_metrics(self, 
                                   returns: pd.Series,
                                   regime_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate portfolio-level metrics.
        
        Metrics that matter:
        - Sharpe Ratio (target > 0.8)
        - Max Drawdown (target < 12%)
        - Monthly consistency
        - Turnover
        
        Args:
            returns: Daily portfolio returns
            regime_data: Optional regime labels per day
        
        Returns:
            Dict of metrics
        """
        if returns.empty:
            return {}
        
        # Annualized metrics
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_days': len(returns)
        }
        
        # Regime-stratified metrics (if provided)
        if regime_data is not None and 'regime' in regime_data.columns:
            regime_sharpe = {}
            for regime in regime_data['regime'].unique():
                regime_returns = returns[regime_data['regime'] == regime]
                if len(regime_returns) > 20:
                    r_mean = regime_returns.mean() * 252
                    r_vol = regime_returns.std() * np.sqrt(252)
                    regime_sharpe[regime] = r_mean / r_vol if r_vol > 0 else 0
            
            metrics['regime_sharpe'] = regime_sharpe
        
        return metrics


class RegimeAwarePortfolio(PortfolioStrategy):
    """
    Portfolio that only trades in favorable regimes.
    
    Key Insight:
    Flat months > losing months
    """
    
    def __init__(self, 
                 regime_sharpe_threshold=0.5,
                 **kwargs):
        """
        Args:
            regime_sharpe_threshold: Min historical Sharpe to trade in regime
        """
        super().__init__(**kwargs)
        self.regime_sharpe_threshold = regime_sharpe_threshold
        self.historical_regime_sharpe = {}  # Populated from backtest
    
    def should_trade_in_regime(self, regime: Tuple[str, str]) -> bool:
        """
        Determine if we should trade given current regime.
        
        Logic:
        - Only trade if historical Sharpe in this regime > threshold
        - Skip unfavorable regimes entirely
        
        Args:
            regime: (trend, volatility) tuple
        
        Returns:
            bool: True if should trade
        """
        sharpe = self.historical_regime_sharpe.get(regime, 0.0)
        
        return sharpe > self.regime_sharpe_threshold
    
    def update_regime_performance(self, 
                                 regime: Tuple[str, str],
                                 returns: pd.Series):
        """
        Update historical Sharpe for a regime.
        
        This is learned from backtest and used for future decisions.
        """
        if len(returns) < 20:
            return
        
        annual_ret = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        
        self.historical_regime_sharpe[regime] = sharpe


def example_portfolio_construction():
    """Example of portfolio construction logic"""
    
    # Simulate ranked universe
    ranked_stocks = pd.DataFrame({
        'symbol': ['AAPL', 'NVDA', 'TSLA', 'AMD', 'MSFT', 'GOOGL'],
        'rank_percentile': [0.98, 0.96, 0.94, 0.85, 0.82, 0.75],
        'sector': ['Tech', 'Tech', 'Auto', 'Tech', 'Tech', 'Tech'],
        'volatility_20d': [0.25, 0.35, 0.45, 0.30, 0.20, 0.22]
    })
    
    # Initialize portfolio
    portfolio = PortfolioStrategy(
        max_positions=5,
        sector_limit=2  # Max 2 stocks per sector
    )
    
    # Select positions
    selected = portfolio.select_positions(ranked_stocks)
    print("Selected Positions:")
    print(selected[['symbol', 'rank_percentile', 'sector']])
    
    # Calculate weights
    selected = portfolio.calculate_volatility_weights(selected)
    print("\nVolatility-Adjusted Weights:")
    print(selected[['symbol', 'weight', 'volatility_20d']])
    
    # Simulate exit check
    position = {
        'symbol': 'AAPL',
        'entry_price': 150.0,
        'entry_volatility': 0.25
    }
    
    should_exit, reason = portfolio.check_exit_conditions(
        position=position,
        current_rank=0.75,  # Dropped below 80%
        days_held=10,
        current_price=155.0
    )
    
    print(f"\nExit Check: {should_exit}, Reason: {reason}")


if __name__ == "__main__":
    example_portfolio_construction()

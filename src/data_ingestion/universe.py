"""
Universe Management Module

Manages the trading universe:
- S&P 500 (US)
- NIFTY 500 (India)
- Quality filters
- Sector mapping
"""

import pandas as pd
import requests
from typing import List, Dict
from datetime import datetime


class UniverseManager:
    """
    Manages the ~1,000 stock trading universe.
    """
    
    def __init__(self):
        self.sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.nifty500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    
    def fetch_sp500_constituents(self) -> pd.DataFrame:
        """
        Fetch S&P 500 constituents.
        
        Returns:
            DataFrame with columns: symbol, name, sector, industry
        """
        try:
            # Method 1: Try Wikipedia with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            tables = pd.read_html(self.sp500_url, storage_options=headers)
            sp500 = tables[0]
            
            # Rename columns
            sp500 = sp500.rename(columns={
                'Symbol': 'symbol',
                'Security': 'name',
                'GICS Sector': 'sector',
                'GICS Sub-Industry': 'industry'
            })
            
        except Exception as e:
            print(f"Wikipedia failed ({e}), using yfinance fallback...")
            
            # Method 2: Use yfinance to get S&P 500 tickers
            import yfinance as yf
            
            sp500_ticker = yf.Ticker("^GSPC")
            # Get top holdings or use hardcoded list
            sp500 = self._get_sp500_fallback()
        
        # Add metadata
        sp500['market'] = 'US'
        sp500['index_name'] = 'S&P 500'
        sp500['active'] = True
        sp500['start_date'] = datetime.now().strftime('%Y-%m-%d')
        sp500['end_date'] = None
        
        return sp500[['symbol', 'name', 'sector', 'industry', 'market', 
                     'index_name', 'active', 'start_date', 'end_date']]
    
    def _get_sp500_fallback(self) -> pd.DataFrame:
        """Fallback S&P 500 top holdings"""
        # Top 100 S&P 500 stocks by market cap
        sp500_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
            "V", "XOM", "WMT", "JPM", "LLY", "PG", "MA", "AVGO", "HD", "CVX",
            "MRK", "ABBV", "PEP", "COST", "KO", "ADBE", "TMO", "MCD", "CSCO", "ACN",
            "ABT", "CRM", "NFLX", "NKE", "LIN", "DHR", "VZ", "TXN", "ORCL", "WFC",
            "PM", "NEE", "DIS", "CMCSA", "INTC", "AMD", "UPS", "RTX", "QCOM", "INTU",
            "HON", "AMGN", "IBM", "BA", "CAT", "GE", "LOW", "SPGI", "SBUX", "AMAT",
            "DE", "BLK", "AXP", "ELV", "BKNG", "GILD", "ADI", "TJX", "MDLZ", "PLD",
            "ISRG", "MMC", "SYK", "VRTX", "ADP", "REGN", "CI", "ZTS", "CB", "MO",
            "PGR", "SO", "DUK", "TGT", "BMY", "LRCX", "BDX", "ETN", "NOC", "BSX",
            "SLB", "SCHW", "ITW", "C", "EOG", "HUM", "PNC", "USB", "MMM", "COP"
        ]
        
        return pd.DataFrame({
            'symbol': sp500_symbols,
            'name': ['Company ' + s for s in sp500_symbols],
            'sector': 'Technology',  # Simplified
            'industry': 'Software'
        })
    
    def fetch_nifty500_constituents(self) -> pd.DataFrame:
        """
        Fetch NIFTY 500 constituents.
        
        Note: NSE website requires headers/cookies. Using fallback list.
        
        Returns:
            DataFrame with columns: symbol, name, sector, industry
        """
        try:
            # Try NSE official (may require authentication)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(self.nifty500_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                from io import StringIO
                nifty500 = pd.read_csv(StringIO(response.text))
                
                # Standardize columns
                nifty500 = nifty500.rename(columns={
                    'Symbol': 'symbol',
                    'Company Name': 'name',
                    'Industry': 'sector'
                })
                
                # Add .NS suffix for Yahoo Finance
                nifty500['symbol'] = nifty500['symbol'] + '.NS'
                
            else:
                # Fallback: Use NIFTY 50 + NIFTY Next 50 for now
                print("NIFTY 500 not accessible, using NIFTY 50 as fallback")
                nifty500 = self._get_nifty50_fallback()
            
            # Add metadata
            nifty500['market'] = 'IN'
            nifty500['index_name'] = 'NIFTY 500'
            nifty500['active'] = True
            nifty500['start_date'] = datetime.now().strftime('%Y-%m-%d')
            nifty500['end_date'] = None
            nifty500['industry'] = nifty500.get('industry', 'UNKNOWN')
            
            return nifty500[['symbol', 'name', 'sector', 'industry', 'market', 
                            'index_name', 'active', 'start_date', 'end_date']]
        
        except Exception as e:
            print(f"Error fetching NIFTY 500: {e}")
            return self._get_nifty50_fallback()
    
    def _get_nifty50_fallback(self) -> pd.DataFrame:
        """Fallback NIFTY 50 list"""
        nifty50_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
            "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TITAN.NS", "BAJFINANCE.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS",
            "HCLTECH.NS", "TECHM.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
            "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "COALINDIA.NS",
            "BAJAJFINSV.NS", "DIVISLAB.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
            "HEROMOTOCO.NS", "HINDALCO.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "BRITANNIA.NS",
            "CIPLA.NS", "APOLLOHOSP.NS", "BPCL.NS", "SHREECEM.NS", "TATACONSUM.NS",
            "UPL.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "ADANIENT.NS"
        ]
        
        return pd.DataFrame({
            'symbol': nifty50_symbols,
            'name': ['Company ' + s.split('.')[0] for s in nifty50_symbols],
            'sector': 'UNKNOWN',
            'industry': 'UNKNOWN'
        })
    
    def create_universe_table(self) -> pd.DataFrame:
        """
        Create combined universe table.
        
        Returns:
            DataFrame with ~1,000 stocks (S&P 500 + NIFTY 500)
        """
        print("Fetching S&P 500...")
        sp500 = self.fetch_sp500_constituents()
        print(f"  Found {len(sp500)} S&P 500 stocks")
        
        print("Fetching NIFTY 500...")
        nifty500 = self.fetch_nifty500_constituents()
        print(f"  Found {len(nifty500)} NIFTY stocks")
        
        # Combine
        universe = pd.concat([sp500, nifty500], ignore_index=True)
        
        print(f"\nTotal Universe: {len(universe)} stocks")
        print(f"  US: {len(universe[universe['market']=='US'])}")
        print(f"  India: {len(universe[universe['market']=='IN'])}")
        
        return universe
    
    def save_universe(self, universe: pd.DataFrame, filepath='data/universe.csv'):
        """Save universe to CSV"""
        universe.to_csv(filepath, index=False)
        print(f"Universe saved to {filepath}")
    
    def load_universe(self, filepath='data/universe.csv') -> pd.DataFrame:
        """Load universe from CSV"""
        return pd.read_csv(filepath)


if __name__ == "__main__":
    manager = UniverseManager()
    
    # Create universe
    universe = manager.create_universe_table()
    
    # Save
    manager.save_universe(universe)
    
    # Show sample
    print("\nSample:")
    print(universe.head(10))
    
    # Sector breakdown
    print("\nSector Breakdown:")
    print(universe.groupby(['market', 'sector']).size())

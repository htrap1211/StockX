# Kite Connect Setup Guide

## Overview
Kite Connect is Zerodha's official API for algorithmic trading in India.

**Advantages over Yahoo Finance**:
- ✅ Official NSE data (no gaps)
- ✅ Sub-second latency
- ✅ Reliable 1-minute bars
- ✅ WebSocket for real-time
- ✅ Historical data API
- ✅ Order placement support

**Cost**: ₹2,000/month

---

## Setup Steps

### 1. Create Zerodha Account
- Visit: https://zerodha.com/
- Complete KYC
- Fund account (minimum ₹2,000)

### 2. Get API Credentials
1. Go to: https://developers.kite.trade/
2. Click "Create new app"
3. Fill in details:
   - App name: StockMind AI
   - Redirect URL: http://localhost:8080
   - Type: Connect
4. Get your **API Key** and **API Secret**

### 3. Set Environment Variables

```bash
# Add to ~/.zshrc or ~/.bashrc
export KITE_API_KEY='your_api_key_here'
export KITE_API_SECRET='your_api_secret_here'

# Reload
source ~/.zshrc
```

### 4. Generate Access Token

```bash
# Run login flow (one-time per day)
python src/intraday/data_ingestion/kite_login.py
```

This will:
1. Open browser for Zerodha login
2. Redirect to localhost with request token
3. Generate access token
4. Save to `.env` file

### 5. Test Connection

```python
from src.intraday.data_ingestion.kite_client import KiteIntradayClient

client = KiteIntradayClient()
df = client.fetch_5min_data('RELIANCE', days_back=7)
print(df.head())
```

---

## Alternative: Continue with Yahoo Finance

**For now**, we can continue using Yahoo Finance for:
- Development
- Backtesting
- Paper trading

**Switch to Kite Connect when**:
- Ready for live trading
- Need real-time data
- Want official NSE data

---

## Current Status

✅ Kite client code ready (`src/intraday/data_ingestion/kite_client.py`)
⚠️ Requires manual setup (account + API credentials)
✅ Fallback to Yahoo Finance works

---

## Next Steps

**Option A: Set up Kite Connect now**
- Follow steps above
- Cost: ₹2,000/month
- Time: 1-2 hours

**Option B: Continue with Yahoo Finance**
- Free
- Good enough for development
- Switch to Kite later for live trading

**Recommendation**: Continue with Yahoo Finance for now, switch to Kite when ready for live trading (after 30 days of paper trading).

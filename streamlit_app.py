"""
StockMind AI - Streamlit Dashboard (Cloud Version)

Standalone version with embedded logic (no external API required)
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Page config
st.set_page_config(
    page_title="StockMind AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #E6EDF3; }
    .stApp { background-color: #0E1117; }
    h1, h2, h3 { color: #E6EDF3; }
    .metric-card {
        background-color: #161B22;
        border: 1px solid #2A2F3A;
        border-radius: 12px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def get_demo_swing_data(market):
    """Generate demo swing trading data"""
    if market == "US":
        symbols = ["AAPL", "NVDA", "MSFT", "GOOGL", "META"]
    else:
        symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]
    
    recommendations = []
    for symbol in symbols:
        rec_type = np.random.choice(["BUY", "WATCH", "AVOID"], p=[0.3, 0.5, 0.2])
        confidence = np.random.uniform(45, 85)
        
        # Get real price from yfinance
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
            else:
                price = 100.0
        except:
            price = 100.0
        
        recommendations.append({
            'symbol': symbol,
            'market': market,
            'recommendation': rec_type,
            'confidence_score': f"{confidence:.1f}%",
            'entry_price': price,
            'stop_loss': price * 0.95,
            'reasoning': ['EMA_50: 0.13', 'ATR_14: 0.09', 'MACD_signal_9: 0.09']
        })
    
    return recommendations

@st.cache_data(ttl=300)
def get_demo_intraday_data():
    """Generate demo intraday signals"""
    signals = [
        {
            'symbol': 'RELIANCE.NS',
            'setup_type': 'ORB',
            'signal': 'LONG',
            'entry_price': 1425.50,
            'current_price': 1428.20,
            'target': 1429.06,
            'stop_loss': 1422.00,
            'confidence': 0.72,
            'status': 'ACTIVE'
        },
        {
            'symbol': 'TCS.NS',
            'setup_type': 'VWAP_REVERSION',
            'signal': 'LONG',
            'entry_price': 3850.00,
            'current_price': 3855.50,
            'target': 3857.70,
            'stop_loss': 3845.00,
            'confidence': 0.68,
            'status': 'ACTIVE'
        },
        {
            'symbol': 'INFY.NS',
            'setup_type': 'ORB',
            'signal': 'LONG',
            'entry_price': 1580.00,
            'current_price': 1584.20,
            'target': 1583.95,
            'stop_loss': 1577.00,
            'confidence': 0.75,
            'status': 'TARGET_HIT'
        }
    ]
    return signals

# Header
st.title("📊 StockMind AI")
st.caption("Institutional-Grade Trading Signals • Demo Version")

# View selector
view = st.radio(
    "Select View",
    ["📊 Swing Trading", "⚡ Intraday Signals"],
    horizontal=True
)

st.divider()

# Swing Trading View
if view == "📊 Swing Trading":
    st.header("Swing Trading Recommendations")
    st.caption("Probability-based signals • 20-day horizon")
    
    market = st.selectbox("Market", ["US", "IN"], index=0)
    
    with st.spinner("Analyzing market data..."):
        recommendations = get_demo_swing_data(market)
    
    if recommendations:
        cols = st.columns(3)
        for idx, rec in enumerate(recommendations):
            with cols[idx % 3]:
                with st.container():
                    # Header
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.subheader(rec['symbol'])
                        st.caption(f"{rec['market']} • SWING")
                    with col_b:
                        badge_color = {
                            'BUY': '🟢',
                            'WATCH': '🟡',
                            'AVOID': '🔴'
                        }.get(rec['recommendation'], '🟡')
                        st.markdown(f"{badge_color} **{rec['recommendation']}**")
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Entry", f"${rec['entry_price']:.2f}")
                    with col2:
                        st.metric("Stop Loss", f"${rec['stop_loss']:.2f}")
                    
                    # Confidence
                    confidence_val = float(rec['confidence_score'].strip('%'))
                    st.progress(confidence_val / 100)
                    st.caption(f"Confidence: {rec['confidence_score']}")
                    
                    # Top signals
                    st.caption("**Top Signals:**")
                    for reason in rec['reasoning'][:3]:
                        st.caption(f"• {reason.split(':')[0]}")
                    
                    st.divider()

# Intraday View
else:
    st.header("Intraday Trading Signals")
    st.caption("Real-time ORB & VWAP setups • NSE Cash")
    
    # Stats
    cols = st.columns(5)
    with cols[0]:
        st.metric("Today's Signals", 3)
    with cols[1]:
        st.metric("Active Trades", 2)
    with cols[2]:
        st.metric("Win Rate", "67%")
    with cols[3]:
        st.metric("Avg Hold", "25m")
    with cols[4]:
        st.metric("Best Setup", "ORB")
    
    st.divider()
    
    signals = get_demo_intraday_data()
    
    if signals:
        cols = st.columns(3)
        for idx, signal in enumerate(signals):
            with cols[idx % 3]:
                with st.container():
                    # Header
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.subheader(signal['symbol'])
                        st.caption(f"{signal['setup_type']} • {signal['signal']}")
                    with col_b:
                        status_emoji = {
                            'ACTIVE': '🔵',
                            'TARGET_HIT': '🟢',
                            'STOPPED': '🔴'
                        }.get(signal['status'], '⚪')
                        st.markdown(f"{status_emoji} **{signal['status']}**")
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Entry", f"₹{signal['entry_price']:.2f}")
                        st.metric("Target", f"₹{signal['target']:.2f}")
                    with col2:
                        st.metric("Current", f"₹{signal['current_price']:.2f}")
                        st.metric("Stop Loss", f"₹{signal['stop_loss']:.2f}")
                    
                    # P&L
                    pnl = ((signal['current_price'] - signal['entry_price']) / signal['entry_price']) * 100
                    pnl_emoji = '🟢' if pnl >= 0 else '🔴'
                    st.markdown(f"{pnl_emoji} **P&L: {pnl:+.2f}%**")
                    
                    # Confidence
                    st.progress(signal['confidence'])
                    st.caption(f"Confidence: {signal['confidence']*100:.0f}%")
                    
                    st.divider()

# Footer
st.divider()
st.caption("© 2026 StockMind AI • Educational purposes only • Not financial advice")
st.caption("⚠️ Demo Version - Using simulated data for demonstration")

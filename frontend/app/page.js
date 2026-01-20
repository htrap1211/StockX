"use client";
import { useEffect, useState } from 'react';
import { fetchRecommendations } from '../utils/api';
import { fetchIntradaySignals, fetchIntradayStats } from '../utils/intraday_api';

export default function Home() {
    const [data, setData] = useState([]);
    const [market, setMarket] = useState("US");
    const [loading, setLoading] = useState(false);
    const [view, setView] = useState("swing"); // 'swing' or 'intraday'
    const [intradaySignals, setIntradaySignals] = useState([]);
    const [intradayStats, setIntradayStats] = useState(null);

    useEffect(() => {
        if (view === "swing") {
            loadData(market);
        } else {
            loadIntradayData();
        }
    }, [market, view]);

    async function loadData(mkt) {
        setLoading(true);
        const recs = await fetchRecommendations(mkt);
        setData(recs);
        setLoading(false);
    }

    async function loadIntradayData() {
        setLoading(true);
        try {
            const [signals, stats] = await Promise.all([
                fetchIntradaySignals(),
                fetchIntradayStats()
            ]);
            setIntradaySignals(signals);
            setIntradayStats(stats);
        } catch (error) {
            console.error('Failed to load intraday data:', error);
        }
        setLoading(false);
    }

    async function loadData(mkt) {
        setLoading(true);
        const recs = await fetchRecommendations(mkt);
        setData(recs);
        setLoading(false);
    }

    return (
        <main style={{ backgroundColor: '#0E1117', minHeight: '100vh', color: '#E6EDF3' }}>
            {/* Header */}
            <nav style={{
                borderBottom: '1px solid #2A2F3A',
                backgroundColor: '#161B22',
                position: 'sticky',
                top: 0,
                zIndex: 50
            }}>
                <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '0 24px', height: '64px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div>
                        <h1 style={{ fontSize: '18px', fontWeight: 600, margin: 0 }}>StockMind AI</h1>
                        <p style={{ fontSize: '12px', color: '#9BA3AF', margin: 0 }}>Institutional-Grade Signals</p>
                    </div>

                    <div style={{ display: 'flex', gap: '12px' }}>
                        {/* View Toggle */}
                        <div style={{ display: 'flex', gap: '4px', backgroundColor: '#0E1117', padding: '4px', borderRadius: '8px', border: '1px solid #2A2F3A' }}>
                            <button
                                onClick={() => setView("swing")}
                                style={{
                                    padding: '8px 16px',
                                    borderRadius: '6px',
                                    fontSize: '14px',
                                    fontWeight: 500,
                                    border: 'none',
                                    cursor: 'pointer',
                                    transition: 'all 150ms',
                                    backgroundColor: view === "swing" ? '#161B22' : 'transparent',
                                    color: view === "swing" ? '#E6EDF3' : '#9BA3AF',
                                    borderBottom: view === "swing" ? '2px solid #2DD4BF' : 'none'
                                }}
                            >
                                📊 Swing
                            </button>
                            <button
                                onClick={() => setView("intraday")}
                                style={{
                                    padding: '8px 16px',
                                    borderRadius: '6px',
                                    fontSize: '14px',
                                    fontWeight: 500,
                                    border: 'none',
                                    cursor: 'pointer',
                                    transition: 'all 150ms',
                                    backgroundColor: view === "intraday" ? '#161B22' : 'transparent',
                                    color: view === "intraday" ? '#E6EDF3' : '#9BA3AF',
                                    borderBottom: view === "intraday" ? '2px solid #FACC15' : 'none'
                                }}
                            >
                                ⚡ Intraday
                            </button>
                        </div>

                        {/* Market Toggle (only for swing) */}
                        {view === "swing" && (
                            <div style={{ display: 'flex', gap: '4px', backgroundColor: '#0E1117', padding: '4px', borderRadius: '8px', border: '1px solid #2A2F3A' }}>
                                <button
                                    onClick={() => setMarket("US")}
                                    style={{
                                        padding: '8px 16px',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        fontWeight: 500,
                                        border: 'none',
                                        cursor: 'pointer',
                                        transition: 'all 150ms',
                                        backgroundColor: market === "US" ? '#161B22' : 'transparent',
                                        color: market === "US" ? '#E6EDF3' : '#9BA3AF',
                                        borderBottom: market === "US" ? '2px solid #60A5FA' : 'none'
                                    }}
                                >
                                    🇺🇸 US
                                </button>
                                <button
                                    onClick={() => setMarket("IN")}
                                    style={{
                                        padding: '8px 16px',
                                        borderRadius: '6px',
                                        fontSize: '14px',
                                        fontWeight: 500,
                                        border: 'none',
                                        cursor: 'pointer',
                                        transition: 'all 150ms',
                                        backgroundColor: market === "IN" ? '#161B22' : 'transparent',
                                        color: market === "IN" ? '#E6EDF3' : '#9BA3AF',
                                        borderBottom: market === "IN" ? '2px solid #60A5FA' : 'none'
                                    }}
                                >
                                    🇮🇳 India
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            </nav>

            <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '32px 24px' }}>
                {/* Page Title */}
                <div style={{ marginBottom: '32px' }}>
                    <h2 style={{ fontSize: '24px', fontWeight: 600, margin: '0 0 8px 0' }}>
                        {view === "swing" ? "Today's AI Picks" : "Intraday Signals"}
                    </h2>
                    <p style={{ color: '#9BA3AF', margin: 0 }}>
                        {view === "swing"
                            ? "Probability-based recommendations • Updated daily"
                            : "Real-time ORB & VWAP setups • NSE Cash"}
                    </p>
                </div>

                {/* Intraday Stats (only for intraday view) */}
                {view === "intraday" && intradayStats && (
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                        gap: '12px',
                        marginBottom: '24px'
                    }}>
                        <StatCard label="Today's Signals" value={intradayStats.today_signals} />
                        <StatCard label="Active Trades" value={intradayStats.active_trades} color="#2DD4BF" />
                        <StatCard label="Win Rate" value={`${(intradayStats.win_rate * 100).toFixed(0)}%`} />
                        <StatCard label="Avg Hold" value={`${intradayStats.avg_hold_time_minutes}m`} />
                        <StatCard label="Best Setup" value={intradayStats.best_setup} color="#FACC15" />
                    </div>
                )}

                {loading ? (
                    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '80px 0' }}>
                        <div style={{
                            width: '40px',
                            height: '40px',
                            border: '3px solid #2A2F3A',
                            borderTop: '3px solid #60A5FA',
                            borderRadius: '50%',
                            animation: 'spin 1s linear infinite'
                        }} />
                        <p style={{ color: '#9BA3AF', marginTop: '16px' }}>Analyzing market data...</p>
                    </div>
                ) : view === "swing" ? (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: '20px' }}>
                        {data.map((row) => (
                            <InstitutionalCard key={row.symbol} data={row} />
                        ))}
                    </div>
                ) : (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(380px, 1fr))', gap: '20px' }}>
                        {intradaySignals.map((signal, idx) => (
                            <IntradaySignalCard key={idx} signal={signal} />
                        ))}
                    </div>
                )}

                {!loading && ((view === "swing" && data.length === 0) || (view === "intraday" && intradaySignals.length === 0)) && (
                    <div style={{ textAlign: 'center', padding: '80px 20px', backgroundColor: '#161B22', borderRadius: '12px', border: '1px solid #2A2F3A' }}>
                        <p style={{ color: '#9BA3AF', fontSize: '16px' }}>
                            {view === "swing" ? "No active signals for this market" : "No intraday setups detected"}
                        </p>
                        <button
                            onClick={() => view === "swing" ? loadData(market) : loadIntradayData()}
                            style={{ marginTop: '12px', color: '#60A5FA', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
                        >
                            Refresh
                        </button>
                    </div>
                )}
            </div>

            {/* Footer */}
            <footer style={{ borderTop: '1px solid #2A2F3A', marginTop: '80px', padding: '24px', textAlign: 'center' }}>
                <p style={{ color: '#9BA3AF', fontSize: '13px', margin: 0 }}>
                    © 2026 StockMind AI • Educational purposes only • Not financial advice
                </p>
            </footer>

            <style jsx>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
        </main>
    );
}

function StatCard({ label, value, color = '#E6EDF3' }) {
    return (
        <div style={{
            backgroundColor: '#161B22',
            border: '1px solid #2A2F3A',
            borderRadius: '8px',
            padding: '16px',
            textAlign: 'center'
        }}>
            <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 8px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                {label}
            </p>
            <p style={{ fontSize: '24px', fontWeight: 600, margin: 0, color: color }}>
                {value}
            </p>
        </div>
    );
}

function IntradaySignalCard({ signal }) {
    const isLong = signal.signal === "LONG";
    const isActive = signal.status === "ACTIVE";
    const isTargetHit = signal.status === "TARGET_HIT";

    const setupColor = signal.setup_type === "ORB" ? '#2DD4BF' : '#FACC15';
    const statusColor = isActive ? '#60A5FA' : isTargetHit ? '#2DD4BF' : '#F87171';

    const pnl = ((signal.current_price - signal.entry_price) / signal.entry_price) * 100;
    const pnlColor = pnl >= 0 ? '#2DD4BF' : '#F87171';

    return (
        <div style={{
            backgroundColor: '#161B22',
            border: '1px solid #2A2F3A',
            borderRadius: '12px',
            padding: '20px',
            transition: 'all 200ms ease',
            cursor: 'pointer'
        }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = setupColor;
                e.currentTarget.style.boxShadow = `0 4px 12px ${setupColor}20`;
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#2A2F3A';
                e.currentTarget.style.boxShadow = 'none';
            }}>

            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                    <h3 style={{ fontSize: '20px', fontWeight: 600, margin: '0 0 4px 0' }}>{signal.symbol}</h3>
                    <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        {signal.setup_type} • {signal.signal}
                    </span>
                </div>
                <div style={{
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    backgroundColor: `${statusColor}20`,
                    color: statusColor,
                    border: `1px solid ${statusColor}40`
                }}>
                    {signal.status}
                </div>
            </div>

            {/* Metrics Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                padding: '16px',
                backgroundColor: '#0E1117',
                borderRadius: '8px',
                marginBottom: '16px',
                border: '1px solid #2A2F3A'
            }}>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Entry</p>
                    <p style={{ fontSize: '16px', fontWeight: 500, margin: 0, fontFamily: 'monospace' }}>₹{signal.entry_price.toFixed(2)}</p>
                </div>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Current</p>
                    <p style={{ fontSize: '16px', fontWeight: 500, margin: 0, fontFamily: 'monospace', color: pnlColor }}>₹{signal.current_price.toFixed(2)}</p>
                </div>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Target</p>
                    <p style={{ fontSize: '16px', fontWeight: 500, margin: 0, color: '#2DD4BF', fontFamily: 'monospace' }}>₹{signal.target.toFixed(2)}</p>
                </div>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Stop Loss</p>
                    <p style={{ fontSize: '16px', fontWeight: 500, margin: 0, color: '#F87171', fontFamily: 'monospace' }}>₹{signal.stop_loss.toFixed(2)}</p>
                </div>
            </div>

            {/* P&L */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px' }}>P&L</span>
                <span style={{ fontSize: '16px', fontWeight: 600, color: pnlColor, fontFamily: 'monospace' }}>
                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}%
                </span>
            </div>

            {/* Confidence */}
            <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Confidence</span>
                    <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: 'monospace' }}>{(signal.confidence * 100).toFixed(0)}%</span>
                </div>
                <div style={{
                    height: '6px',
                    backgroundColor: '#2A2F3A',
                    borderRadius: '3px',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        height: '100%',
                        width: `${signal.confidence * 100}%`,
                        backgroundColor: setupColor,
                        transition: 'width 1s ease-out'
                    }} />
                </div>
            </div>
        </div>
    );
}

function InstitutionalCard({ data }) {
    const isBuy = data.recommendation === "BUY";
    const isAvoid = data.recommendation === "AVOID";
    const isWatch = data.recommendation === "WATCH";

    // Determine signal color
    const signalColor = isBuy ? '#2DD4BF' : isAvoid ? '#F87171' : '#FACC15';
    const signalBg = isBuy ? 'rgba(45, 212, 191, 0.1)' : isAvoid ? 'rgba(248, 113, 113, 0.1)' : 'rgba(250, 204, 21, 0.1)';

    // Parse confidence percentage
    const confidenceNum = parseFloat(data.confidence_score);
    const confidenceLevel = confidenceNum > 60 ? 'High' : confidenceNum > 40 ? 'Medium' : 'Low';

    return (
        <div style={{
            backgroundColor: '#161B22',
            border: '1px solid #2A2F3A',
            borderRadius: '12px',
            padding: '20px',
            transition: 'all 200ms ease',
            cursor: 'pointer'
        }}
            onMouseEnter={(e) => {
                e.currentTarget.style.borderColor = '#60A5FA';
                e.currentTarget.style.boxShadow = '0 4px 12px rgba(96, 165, 250, 0.1)';
            }}
            onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = '#2A2F3A';
                e.currentTarget.style.boxShadow = 'none';
            }}>

            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                <div>
                    <h3 style={{ fontSize: '20px', fontWeight: 600, margin: '0 0 4px 0' }}>{data.symbol}</h3>
                    <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                        {data.market} • SWING
                    </span>
                </div>
                <div style={{
                    padding: '4px 10px',
                    borderRadius: '6px',
                    fontSize: '11px',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                    letterSpacing: '0.5px',
                    backgroundColor: signalBg,
                    color: signalColor,
                    border: `1px solid ${signalColor}40`
                }}>
                    {data.recommendation}
                </div>
            </div>

            {/* Metrics Grid */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                padding: '16px',
                backgroundColor: '#0E1117',
                borderRadius: '8px',
                marginBottom: '16px',
                border: '1px solid #2A2F3A'
            }}>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Entry</p>
                    <p style={{ fontSize: '18px', fontWeight: 500, margin: 0, fontFamily: 'monospace' }}>${data.entry_price?.toFixed(2)}</p>
                </div>
                <div>
                    <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 4px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Stop Loss</p>
                    <p style={{ fontSize: '18px', fontWeight: 500, margin: 0, color: '#F87171', fontFamily: 'monospace' }}>${data.stop_loss?.toFixed(2)}</p>
                </div>
            </div>

            {/* Confidence */}
            <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Confidence</span>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 500, color: '#60A5FA' }}>{confidenceLevel}</span>
                        <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: 'monospace' }}>{data.confidence_score}</span>
                    </div>
                </div>
                <div style={{
                    height: '6px',
                    backgroundColor: '#2A2F3A',
                    borderRadius: '3px',
                    overflow: 'hidden'
                }}>
                    <div style={{
                        height: '100%',
                        width: data.confidence_score,
                        backgroundColor: '#60A5FA',
                        transition: 'width 1s ease-out'
                    }} />
                </div>
            </div>

            {/* Risk Badge */}
            <div style={{ marginBottom: '16px' }}>
                <span style={{ fontSize: '11px', color: '#9BA3AF', textTransform: 'uppercase', letterSpacing: '0.5px', marginRight: '8px' }}>Risk:</span>
                <span style={{
                    padding: '2px 8px',
                    borderRadius: '4px',
                    fontSize: '11px',
                    fontWeight: 500,
                    backgroundColor: confidenceNum > 60 ? '#2A2F3A' : confidenceNum > 40 ? 'rgba(250, 204, 21, 0.1)' : 'rgba(248, 113, 113, 0.1)',
                    color: confidenceNum > 60 ? '#9BA3AF' : confidenceNum > 40 ? '#FACC15' : '#F87171'
                }}>
                    {confidenceNum > 60 ? 'Low' : confidenceNum > 40 ? 'Medium' : 'High'}
                </span>
            </div>

            {/* Top Reasons */}
            <div>
                <p style={{ fontSize: '11px', color: '#9BA3AF', margin: '0 0 8px 0', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Top Signals</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {data.reasoning.slice(0, 3).map((r, i) => (
                        <div key={i} style={{
                            fontSize: '12px',
                            color: '#E6EDF3',
                            padding: '6px 10px',
                            backgroundColor: '#0E1117',
                            borderRadius: '6px',
                            border: '1px solid #2A2F3A',
                            fontFamily: 'monospace'
                        }}>
                            • {r.split(":")[0]}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

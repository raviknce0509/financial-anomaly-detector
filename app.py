import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data import fetch_stock_data
from src.features import engineer_features
from src.detector import detect_anomalies, get_anomaly_summary

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Anomaly Detector",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Market Anomaly Detector")
st.markdown("*Isolation Forest model detecting anomalous trading patterns in real-time market data*")

# ── Sidebar controls ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker", value="AAPL",
    help="Enter any valid ticker: AAPL, TSLA, MSFT, SPY"
).upper()

period = st.sidebar.selectbox(
    "Time Period",
    ["3mo", "6mo", "1y", "2y"],
    index=2
)

contamination = st.sidebar.slider(
    "Anomaly Sensitivity",
    min_value=0.01, max_value=0.10,
    value=0.05, step=0.01,
    help="Expected % of anomalous days"
)

run_btn = st.sidebar.button("🔍 Detect Anomalies", type="primary")

# ── Main logic ───────────────────────────────────────────────────
if run_btn:
    with st.spinner(f"Fetching {ticker} data and running anomaly detection..."):
        try:
            # Pipeline
            raw_df      = fetch_stock_data(ticker, period)
            feature_df  = engineer_features(raw_df)
            result_df, model, scaler = detect_anomalies(
                feature_df, contamination
            )
            summary = get_anomaly_summary(result_df)

            # ── Summary metrics ──────────────────────────────────
            st.subheader(f"📊 {ticker} — Anomaly Detection Results")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Trading Days", summary['total_days'])
            col2.metric("Anomaly Days", summary['anomaly_days'],
                       delta=summary['anomaly_rate'])
            col3.metric("Extreme Events", summary['extreme_days'])
            col4.metric("Latest Signal", summary['latest_signal'])

            st.divider()

            # ── Price chart with anomalies ───────────────────────
            st.subheader("📉 Price Chart with Anomaly Signals")

            normal  = result_df[result_df['is_anomaly'] == 0]
            anomaly = result_df[result_df['is_anomaly'] == 1]
            extreme = result_df[result_df['risk_level'] == 'EXTREME']

            fig = go.Figure()

            # Price line
            fig.add_trace(go.Scatter(
                x=result_df.index, y=result_df['Close'],
                mode='lines', name='Close Price',
                line=dict(color='#2196F3', width=1.5)
            ))

            # Bollinger Bands
            fig.add_trace(go.Scatter(
                x=result_df.index, y=result_df['bb_upper'],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=result_df.index, y=result_df['bb_lower'],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                opacity=0.5
            ))

            # Anomaly markers
            fig.add_trace(go.Scatter(
                x=anomaly.index, y=anomaly['Close'],
                mode='markers', name='Anomaly',
                marker=dict(color='orange', size=8, symbol='circle')
            ))

            # Extreme markers
            fig.add_trace(go.Scatter(
                x=extreme.index, y=extreme['Close'],
                mode='markers', name='Extreme',
                marker=dict(color='red', size=12, symbol='x')
            ))

            fig.update_layout(
                height=500,
                template='plotly_dark',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                legend=dict(orientation='h', y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Return Z-Score chart ─────────────────────────────
            st.subheader("📊 Return Z-Score (Anomaly Signal Strength)")

            fig2 = px.bar(
                result_df.tail(60),
                x=result_df.tail(60).index,
                y='return_zscore',
                color='risk_level',
                color_discrete_map={
                    'NORMAL': '#4CAF50',
                    'ANOMALY': '#FF9800',
                    'EXTREME': '#F44336'
                },
                template='plotly_dark'
            )
            fig2.add_hline(y=2, line_dash="dash",
                          line_color="orange", opacity=0.7)
            fig2.add_hline(y=-2, line_dash="dash",
                          line_color="orange", opacity=0.7)
            fig2.add_hline(y=3, line_dash="dash",
                          line_color="red", opacity=0.7)
            fig2.add_hline(y=-3, line_dash="dash",
                          line_color="red", opacity=0.7)
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

            # ── Anomaly table ────────────────────────────────────
            st.subheader("🚨 Detected Anomaly Events")
            anomaly_table = result_df[result_df['is_anomaly'] == 1][[
                'Close', 'daily_return', 'return_zscore',
                'volume_spike', 'risk_level'
            ]].copy()
            anomaly_table['daily_return'] = (
                anomaly_table['daily_return'] * 100
            ).round(2).astype(str) + '%'
            anomaly_table['return_zscore'] = (
                anomaly_table['return_zscore'].round(2)
            )
            anomaly_table['volume_spike'] = (
                anomaly_table['volume_spike'].round(2)
            )
            st.dataframe(anomaly_table, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.info("Try a different ticker or time period.")

else:
    # Landing state
    st.info("👈 Enter a stock ticker and click **Detect Anomalies** to begin.")
    st.markdown("""
    ### How It Works
    1. **Data:** Pulls real-time stock data via Yahoo Finance API
    2. **Features:** Engineers 7 financial features including Bollinger Bands, 
       Z-scores, volume spikes
    3. **Model:** Isolation Forest detects statistically unusual trading days
    4. **Output:** Visual chart with anomaly markers + risk classification table
    
    ### Example Tickers to Try
    `AAPL` `TSLA` `MSFT` `SPY` `NVDA` `JPM` `BTC-USD`
    """)
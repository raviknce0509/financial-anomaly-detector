---
title: Financial Anomaly Detector
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
short_description: Detect anomalous trading patterns in real-time stock data
license: mit
---

# Financial Market Anomaly Detector

Detects anomalous trading patterns in real-time stock market data using an Isolation Forest model.

## Features
- Real-time stock data via Yahoo Finance
- 7 engineered financial features (Bollinger Bands, Z-scores, volume spikes)
- Isolation Forest anomaly detection
- Interactive charts with anomaly markers and risk classification

## Usage
Enter any valid stock ticker (AAPL, TSLA, MSFT, SPY, BTC-USD) and click **Detect Anomalies**.

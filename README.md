# Adaptive Market Regime Intelligence Engine

An autonomous market intelligence system designed to identify changing market regimes and provide regime-aware context for adaptive trading decisions.

## Current Status

### Completed
- Historical NIFTY 50 market data ingestion
- Data preprocessing and cleaning
- Feature engineering
- Market-state feature matrix generation
- Exploratory visualization of market features

### Current Feature Families

- Short-term, medium-term and long-term returns
- Moving averages
- Price-to-moving-average ratios
- Rolling volatility
- RSI-based momentum
- Intraday price ranges
- Trading volume features

## Architecture

```text
Market Data
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Feature Matrix
    ↓
Market Regime Detection
    ↓
Regime-Aware Decision Agent
    ↓
Risk / Permission Gate
    ↓
Paper Trading
    ↓
Evaluation
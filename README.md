## Project Overview

This notebook demonstrates how a rule-based trading strategy can be designed under realistic market assumptions, explicitly accounting for key characteristics of financial time-series such as non-IID behavior, volatility clustering, and regime shifts.

Rather than maximizing in-sample performance, the strategy emphasizes robustness, interpretability, and economic intuition.

---

## Data & Feature Engineering

- Loads `ETHUSDT.csv` containing timestamped open, high, low, close, and volume data  
- Sorts observations chronologically, checks data integrity, and computes descriptive statistics  
- Transforms raw prices into more stationary representations (log returns and volatility-adjusted features)

### Feature Construction

Features are constructed using TA-Lib and custom transformations to capture different market dimensions:

- **Trend features**: SMA/EMA across multiple horizons, moving-average slopes, distance to long-term trend  
- **Momentum features**: MACD, RSI (multiple windows), Stochastic Oscillator, Rate of Change, Momentum  
- **Volatility features**: ATR, Bollinger Bands, rolling volatility measures  
- **Volume features**: Volume moving averages, OBV, volume change ratios  
- **Custom features**: Normalized distances to moving averages and Bollinger Bands  

These features are designed to reflect trend persistence, momentum strength, volatility regimes, and market participation, rather than raw price levels.

---

## Strategy Design & Trading Logic

The strategy is implemented as an `AdvancedETHStrategy` (Backtesting.py style) with configurable parameters for:

- Moving-average horizons  
- RSI thresholds  
- ATR-based stop-loss and take-profit multipliers  
- Position sizing rules  

### Entry Logic (Long-Only)

Positions are initiated only when multiple conditions jointly confirm a favorable regime:

- Short-term trend exceeds long-term trend (fast SMA above slow SMA)  
- Price trades above the long-term (200-period) moving average  
- Momentum confirmation via RSI above a threshold  
- MACD above its signal line  

An optional mean-reversion entry is allowed when price touches Bollinger Band extremes within an established uptrend, capturing temporary pullbacks rather than counter-trend moves.

### Exit Logic & Risk Management

Positions are exited when trend or momentum conditions deteriorate, including:

- RSI overbought signals  
- Moving-average or MACD reversals  
- Bollinger Band exhaustion  

Risk is controlled through dynamic ATR-based stop-loss and take-profit levels, allowing exits to adapt to changing volatility conditions.

---

## Backtesting & Performance Evaluation

Backtesting is conducted using Backtesting.py with the following assumptions:

- Initial capital: 10,000 USDT  
- Transaction cost: 0.1% per trade  
- Exclusive orders to prevent overlapping positions  

Reported performance metrics include:

- Total and annualized return  
- Buy-and-hold benchmark return  
- Sharpe ratio  
- Maximum drawdown  
- Win rate and profit factor  
- Average trade return and trade duration  

The strategy is explicitly evaluated against the following objectives:

- Sharpe ratio greater than 0.3  
- Annualized return greater than 15%  
- Outperformance versus buy-and-hold over the same period  

---

## Visualization & Trade-Level Analysis

The notebook provides extensive visual diagnostics, including:

- Price series with moving averages  
- RSI, MACD, and volume indicators  
- Equity curve and drawdown trajectory  
- Trade return distributions and win/loss breakdowns  
- Strategy versus buy-and-hold cumulative returns  
- Monthly return heatmaps  

An interactive HTML backtest chart is generated for detailed inspection.

Trade-level analysis includes:

- Win/loss counts and ratios  
- Average and maximum gains and losses  
- Expectancy per trade  
- Examples of best and worst trades  

---

## Robustness Checks & Risk Metrics

To assess stability beyond a single backtest:

- Parameter optimization is performed over MA, RSI, and ATR hyperparameters using Sharpe ratio as the objective  
- Walk-forward analysis is conducted with rolling train/test windows to evaluate performance consistency across time  

Additional risk metrics include:

- Daily and annualized volatility  
- Sortino and Calmar ratios  
- Value-at-Risk (VaR) and Conditional VaR (CVaR)  
- Rolling Sharpe and volatility  
- Drawdown statistics and maximum consecutive losses  

These analyses aim to distinguish structural trading edge from potential overfitting.

---

## Notes

- The strategy is designed for research and educational purposes only  
- Results are sensitive to market regime and transaction costs  
- No claim is made regarding future performance

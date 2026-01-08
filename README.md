# ETHUSDT Trading Strategy

The notebook implements and evaluates a full ETHUSDT trading strategy on 30-minute OHLCV data, from raw data loading through strategy design, backtesting, risk analysis, and validation.

---

## Data and Features

### Data Preparation

Loads ETHUSDT.csv (timestamp, open, high, low, close, volume), sorts by time, checks missing values, and computes basic descriptive statistics.

### Feature Engineering

Builds many technical indicators with TA-Lib: SMAs/EMAs (trend), MACD, multiple RSIs, Stochastic, ROC, momentum, Bollinger Bands, ATR, ADX, volume averages, OBV, volatility and return features, and custom distances to MAs/Bollinger Bands.

---

## Strategy Logic

Defines an AdvancedETHStrategy in backtesting.py style with parameters for MA periods, RSI thresholds, ATR stop/target multipliers, and position size.

Long entries require: fast SMA above slow SMA, price above 200-SMA, RSI above a threshold, MACD above its signal, plus an optional mean-reversion variant using Bollinger Bands in an uptrend; exits use RSI overbought, MA/MACD reversal, or Bollinger extremes, with ATR-based stop-loss and take-profit.

---

## Backtest and Performance

Prepares data for Backtest, runs the strategy with initial capital 10,000 USDT, 0.1% commission, and exclusive orders, and prints the full stats table.

Computes and reports key metrics: total and annualized return, buy-and-hold return, Sharpe ratio, max drawdown, win rate, profit factor, average trade, and trade durations, and checks whether Sharpe > 0.3, annual return > 15%, and return beats buy-and-hold.

---

## Visualization and Trade Analysis

Produces plots for price with MAs, RSI, MACD, volume, the equity curve, drawdowns, trade-return histogram, strategy vs buy-and-hold returns, monthly return heatmap, and win/loss bar charts, plus an interactive HTML backtest chart.

Analyzes individual trades: counts wins/losses, computes average and max win/loss, win–loss ratio, expectancy per trade, shows sample trades, and highlights best and worst trades.

---

## Robustness Checks and Risk Metrics

Runs parameter optimization over MA/RSI/ATR hyperparameters to maximize Sharpe, prints the best parameter set and optimized performance.

Performs a walk-forward analysis: multiple rolling train/test windows on the time axis, collecting Sharpe, return, drawdown, win rate, and trades per period, and plots these over time.

Computes risk metrics: daily and annualized volatility, Sortino and Calmar ratios, VaR/CVaR on returns, drawdown statistics, rolling Sharpe and volatility, and max consecutive losses, with corresponding charts.

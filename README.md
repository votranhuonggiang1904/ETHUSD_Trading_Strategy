# ETHUSDT Trading Strategy

The notebook implements and evaluates a full ETHUSDT trading strategy using 30-minute OHLCV data, covering the entire workflow from raw data loading and feature engineering to strategy design, backtesting, risk analysis, and robustness validation.

---

## Data and Features

### Data Preparation

- Loads `ETHUSDT.csv` containing timestamp, open, high, low, close, and volume data  
- Sorts the data chronologically to preserve time-ordering  
- Checks for missing values and data consistency  
- Computes basic descriptive statistics to understand price behavior and volatility characteristics  

### Feature Engineering

A wide range of technical and statistical features is constructed using TA-Lib and custom transformations, including:

- **Trend indicators**:  
  Simple and exponential moving averages (SMAs and EMAs) across multiple horizons to capture short-, medium-, and long-term trends  

- **Momentum indicators**:  
  MACD, multiple RSI windows, Stochastic Oscillator, Rate of Change (ROC), and momentum indicators to measure trend strength and continuation  

- **Volatility indicators**:  
  Bollinger Bands, Average True Range (ATR), rolling volatility, and return-based volatility measures  

- **Volume-based indicators**:  
  Volume moving averages and On-Balance Volume (OBV) to capture participation and confirmation effects  

- **Custom features**:  
  Normalized distances of price to moving averages and Bollinger Bands, as well as return-based and volatility-adjusted features  

These features are designed to represent different dimensions of market behavior rather than raw price levels.

---

## Strategy Logic

### Strategy Structure

The trading logic is implemented as an `AdvancedETHStrategy` following the Backtesting.py framework.  
The strategy is parameterized by:

- Moving-average periods  
- RSI threshold levels  
- ATR-based stop-loss and take-profit multipliers  
- Position sizing rules  

### Entry Logic (Long-Only)

Long positions are initiated when a set of trend and momentum conditions jointly indicate a favorable regime:

- Fast moving average above slow moving average  
- Price trading above the long-term (200-period) moving average  
- RSI above a predefined threshold  
- MACD above its signal line  

In addition to trend-following entries, an optional mean-reversion variant is included.  
This allows entries near Bollinger Band lower extremes **only when the broader trend remains bullish**, aiming to capture temporary pullbacks within an uptrend.

### Exit Logic and Risk Management

Positions are exited based on a combination of momentum deterioration and trend reversal signals, including:

- RSI entering overbought territory  
- Moving-average or MACD reversals  
- Price reaching Bollinger Band extremes  

Risk management is handled through dynamic ATR-based stop-loss and take-profit levels, allowing exits to adjust automatically to changing volatility regimes.

---

## Backtesting and Performance Evaluation

### Backtesting Setup

- Backtests are conducted using the Backtesting.py framework  
- Initial capital is set to 10,000 USDT  
- Transaction cost is fixed at 0.1% per trade  
- Exclusive orders are enforced to prevent overlapping positions  

### Performance Metrics

The following performance statistics are computed and reported:

- Total return and annualized return  
- Buy-and-hold return over the same period  
- Sharpe ratio  
- Maximum drawdown  
- Win rate  
- Profit factor  
- Average trade return  
- Trade duration statistics  

The strategy performance is evaluated against explicit targets:

- Sharpe ratio greater than 0.3  
- Annualized return greater than 15%  
- Performance exceeding the buy-and-hold benchmark  

---

## Visualization and Trade Analysis

### Visual Diagnostics

The notebook produces a comprehensive set of visualizations, including:

- Price charts with moving averages  
- RSI, MACD, and volume indicators  
- Equity curve and drawdown series  
- Trade return histograms  
- Strategy versus buy-and-hold cumulative returns  
- Monthly return heatmaps  
- Win and loss distribution bar charts  

An interactive HTML backtest chart is also generated for detailed inspection of trades and equity evolution.

### Trade-Level Analysis

Individual trades are analyzed in detail by:

- Counting winning and losing trades  
- Computing average and maximum wins and losses  
- Calculating win–loss ratios and trade expectancy  
- Displaying sample trades  
- Highlighting the best and worst trades  

---

## Robustness Checks and Risk Metrics

### Parameter Optimization

- Performs parameter optimization over moving-average periods, RSI thresholds, and ATR multipliers  
- Optimization objective is the Sharpe ratio  
- Reports the best parameter combination and the corresponding optimized performance  

### Walk-Forward Analysis

- Conducts walk-forward analysis using multiple rolling train/test windows  
- Collects Sharpe ratio, return, drawdown, win rate, and number of trades for each window  
- Visualizes the evolution of these metrics over time  

### Risk Metrics

Additional risk measures are computed and analyzed, including:

- Daily and annualized volatility  
- Sortino ratio and Calmar ratio  
- Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)  
- Rolling Sharpe ratio and rolling volatility  
- Drawdown statistics and maximum consecutive losses  

These analyses are intended to assess strategy stability and distinguish structural performance from potential overfitting.

---

## Notes

- The strategy is developed for research and educational purposes  
- Results are sensitive to market regimes and transaction cost assumptions  
- No claims are made regarding future or real-world trading performance

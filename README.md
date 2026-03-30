# ML-Based Quantitative Trading Backtester

A Python framework for comparing **Logistic Regression** and **XGBoost** models in predicting short‑term stock price movements. The system includes automated data ingestion, technical indicator engineering, risk‑managed strategy execution, and interactive performance visualization.

---

## 🚀 Features

- **Automated Data Pipeline**: Fetch historical data for any ticker (default: `RELIANCE.NS`) using `yfinance`.
- **Feature Engineering**: Generates 15+ technical indicators including RSI, MACD, Bollinger Bands, and trend‑strength ratios using the `ta` library.
- **ML Model Comparison**:
  - **Logistic Regression**: Benchmarked with standardized features.
  - **XGBoost**: Optimized with tuned hyperparameters (e.g., `subsample`, `reg_lambda`, `max_depth`).
- **Vectorized Backtesting**: Simulates trading strategies with a volatility‑adjusted position‑sizing logic.
- **Interactive Analytics**: Visualizes equity curves, market benchmarks, and drawdowns using `Plotly`.

---

## 📁 Project Structure

```text
ML-Trading-Backtester/
├── main.py               # Entry point orchestrating data, training, and backtesting
├── requirements.txt      # Python dependencies
└── src/
    ├── data_handler.py   # Data fetching, feature engineering, and target labeling
    ├── model_builder.py  # Training logic for Logistic Regression and XGBoost
    └── backtester.py     # Strategy execution, metrics (CAGR, Sharpe, Max DD), and plotting
```

---

## 🛠️ Technical Stack

- **Language**: Python
- **Data Analysis**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `xgboost`
- **Financial Indicators**: `ta` (Technical Analysis Library)
- **Visualization**: `plotly`
- **Data Source**: `yfinance`

---

## 📊 Strategy Logic

The strategy predicts short‑term direction (up/down) using a **binary classifier output probability**.  
Buy signals are triggered when the predicted probability exceeds a **threshold of `0.52`**.

To manage risk, the framework uses **volatility‑targeted position sizing**:

\[
\text{Position} = \min\left(1, \frac{\text{Target\_Vol}}{\text{Rolling\_Vol}_{20d}}\right)
\]

This scales down exposure during periods of high 20‑day rolling volatility.

---

## 🏁 Getting Started

1. **Clone the repository**:

```bash
git clone https://github.com/patidarmithil/ML-Trading-Backtester.git
cd ML-Trading-Backtester
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the pipeline**:

```bash
python main.py
```

This will:
- Download historical data for the default ticker (`RELIANCE.NS`).
- Engineer technical indicators.
- Train both Logistic Regression and XGBoost models.
- Backtest the strategies and generate interactive `Plotly` charts.

---

## 📈 Example Output

After running the backtester, you will see:
- Equity curves for both models versus a buy‑and‑hold benchmark.
- Cumulative returns, CAGR, Sharpe ratio, and maximum drawdown metrics.
- Interactive plots to inspect performance across time and volatility regimes.

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

PROB_THRESHOLD = 0.52
TARGET_VOL = 0.01

def apply_strategy(df, prob_col):
    df = df.copy()
    df["signal"] = (df[prob_col] > PROB_THRESHOLD).astype(int)
    df["position"] = TARGET_VOL / df["vol_20"]
    df["position"] = df["position"].clip(0, 1)
    df["final_position"] = df["signal"] * df["position"]
    return df

def backtest(df):
    df = df.copy()
    df["strategy_ret"] = df["final_position"].shift(1) * df["ret_1d"]
    df["strategy_ret"] = df["strategy_ret"].fillna(0)
    df["equity"] = (1 + df["strategy_ret"]).cumprod()
    df["buy_hold"] = (1 + df["ret_1d"]).cumprod()
    df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
    return df

def performance_metrics(df):
    ret = df["strategy_ret"]
    cagr = df["equity"].iloc[-1] ** (252 / len(df)) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    max_dd = df["drawdown"].min()
    
    return {
        "CAGR": round(cagr, 4),
        "Sharpe": round(sharpe, 2),
        "Max Drawdown": round(max_dd, 4)
    }

def model_accuracy(df, prob_col):
    y_true = df["target"]
    y_pred = (df[prob_col] > PROB_THRESHOLD).astype(int)
    return round(accuracy_score(y_true, y_pred) * 100, 2)

def plot_results(test_lr, test_xgb, title="Actual Market vs ML Strategies"):
    # Ensure market price is calculated
    test_lr["market_price"] = test_lr["Close"] / test_lr["Close"].iloc[0]
    test_xgb["market_price"] = test_xgb["Close"] / test_xgb["Close"].iloc[0]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.05
    )

    # Market Price
    fig.add_trace(go.Scatter(x=test_lr.index, y=test_lr["market_price"], name="Actual Market Price", line=dict(color="black", width=3)), row=1, col=1)
    
    # Strategies
    fig.add_trace(go.Scatter(x=test_lr.index, y=test_lr["equity"], name="Logistic Regression Strategy", line=dict(width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_xgb.index, y=test_xgb["equity"], name="XGBoost Strategy", line=dict(width=2)), row=1, col=1)
    
    # Drawdown (Using XGBoost as example)
    fig.add_trace(go.Scatter(x=test_xgb.index, y=test_xgb["drawdown"], name="XGBoost Drawdown", fill="tozeroy"), row=2, col=1)

    fig.update_layout(title=title, height=750, hovermode="x unified")
    fig.show()

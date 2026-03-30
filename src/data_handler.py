import yfinance as yf
import pandas as pd
import numpy as np
import ta

def load_data(ticker="RELIANCE.NS", start="2010-01-01", end="2024-01-01"):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

def create_features(df):
    df = df.copy()
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)

    # Returns
    df["ret_1d"]  = close.pct_change()
    df["ret_5d"]  = close.pct_change(5)
    df["ret_10d"] = close.pct_change(10)
    df["ret_20d"] = close.pct_change(20)

    # Moving Averages & Trends
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_ratio"] = close / df["sma_50"]
    df["trend_strength"] = df["sma_20"] / df["sma_50"]

    # Volatility
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["atr"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Momentum
    df["rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_5"] = ta.momentum.RSIIndicator(close, window=5).rsi()
    df["rsi_10"] = ta.momentum.RSIIndicator(close, window=10).rsi()
    df["macd"] = ta.trend.MACD(close).macd_diff()
    df["bb_width"] = ta.volatility.BollingerBands(close).bollinger_wband()
    df["volume_change"] = df["Volume"].pct_change(5)

    return df.dropna()

def create_target(df, horizon=10, min_move=0.002):
    df = df.copy()
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    df["target"] = np.where(
        future_ret > min_move, 1,
        np.where(future_ret < -min_move, 0, np.nan)
    )
    return df.dropna()

def prepare_data(ticker="RELIANCE.NS", split_date="2021-01-01"):
    df = load_data(ticker=ticker)
    df = create_features(df)
    df = create_target(df)
    
    train = df[df.index < split_date].copy()
    test  = df[df.index >= split_date].copy()
    
    features = [
        "ret_5d", "ret_10d", "ret_20d", "sma_ratio", "trend_strength",
        "vol_20", "atr", "rsi", "rsi_5", "rsi_10", "macd", "bb_width", "volume_change"
    ]
    
    # Clean infinities and NaNs
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    return train, test, features

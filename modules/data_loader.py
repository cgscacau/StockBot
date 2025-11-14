import pandas as pd
import numpy as np
import yfinance as yf
import ta


# ========================= ATR =========================
def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr


# ========================= HISTÓRICO =========================
def get_history(ticker: str, period: str = "1y", interval: str = "1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.dropna(inplace=True)
    return df


# ========================= INDICADORES =========================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df

    # MACD
    try:
        macd = ta.trend.MACD(df["Close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    except Exception:
        df["macd"] = np.nan
        df["macd_signal"] = np.nan
        df["macd_hist"] = np.nan

    # RSI
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    except Exception:
        df["rsi"] = np.nan

    # ADX / DMI
    try:
        adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
        df["adx"] = adx.adx()
        df["+di"] = adx.adx_pos()
        df["-di"] = adx.adx_neg()
    except Exception:
        df["adx"] = np.nan
        df["+di"] = np.nan
        df["-di"] = np.nan

    # KST
    try:
        kst = ta.trend.KSTIndicator(df["Close"])
        df["kst"] = kst.kst()
        df["kst_signal"] = kst.kst_sig()
    except Exception:
        df["kst"] = np.nan
        df["kst_signal"] = np.nan

    # EMAs
    df["ema_fast"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=21, adjust=False).mean()

    # ATR
    df["atr"] = compute_atr(df)

    df.dropna(inplace=True)
    return df

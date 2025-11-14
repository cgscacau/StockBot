import numpy as np
import pandas as pd
from itertools import product
from .data_loader import compute_atr


# ========================= GERA SINAIS =========================
def generate_rule_signals(df, fast=9, slow=21, rr=2.0, atr_mult=1.0):
    df = df.copy()
    if df.empty:
        return df

    df["ema_fast"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=slow, adjust=False).mean()
    df["atr"] = compute_atr(df)

    df["signal"] = 0
    df.loc[(df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)), "signal"] = 1
    df.loc[(df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)), "signal"] = -1

    df["direction"] = "NEUTRAL"
    df.loc[df["signal"] == 1, "direction"] = "BUY"
    df.loc[df["signal"] == -1, "direction"] = "SELL"

    df["entry"] = df["Close"].where(df["signal"] != 0, np.nan)

    # Stops e Targets
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan

    # BUY
    buy = df["signal"] == 1
    df.loc[buy, "stop_loss"] = df["Close"] - atr_mult * df["atr"]
    df.loc[buy, "take_profit"] = df["Close"] + rr * (df["Close"] - df["stop_loss"])

    # SELL
    sell = df["signal"] == -1
    df.loc[sell, "stop_loss"] = df["Close"] + atr_mult * df["atr"]
    df.loc[sell, "take_profit"] = df["Close"] - rr * (df["stop_loss"] - df["Close"])

    return df


# ========================= BACKTEST =========================
def backtest(df, capital=100_000):
    df = df.copy()
    if df.empty:
        return {"return": 0, "max_dd": 0, "equity": []}

    df["ret"] = df["Close"].pct_change().fillna(0)

    pos = 0
    eq = capital
    equity_curve = []

    for i in range(1, len(df)):
        if df["signal"].iloc[i] == 1:
            pos = 1
        elif df["signal"].iloc[i] == -1:
            pos = -1

        eq *= (1 + pos * df["ret"].iloc[i])
        equity_curve.append(eq)

    equity_curve = np.array(equity_curve)
    if len(equity_curve) == 0:
        return {"return": 0, "max_dd": 0, "equity": []}

    ret = (equity_curve[-1] / capital) - 1
    max_dd = ((equity_curve - equity_curve.max()) / equity_curve.max()).min()

    return {
        "return": ret,
        "max_dd": max_dd,
        "equity": equity_curve
    }


# ========================= OTIMIZAÇÃO =========================
def optimize(df, fast_list, slow_list, rr_list, atr_list, progress=None):
    combos = [
        (f, s, rr, am)
        for f in fast_list
        for s in slow_list
        for rr in rr_list
        for am in atr_list
        if f < s
    ]

    best_score = -1e9
    best_params = None
    best_res = None

    total = len(combos)
    for i, (f, s, rr, am) in enumerate(combos, start=1):
        df_sig = generate_rule_signals(df, fast=f, slow=s, rr=rr, atr_mult=am)
        res = backtest(df_sig)
        score = res["return"] - abs(res["max_dd"])

        if score > best_score:
            best_score = score
            best_params = (f, s, rr, am)
            best_res = res

        if progress:
            progress.progress(i / total)

    if progress:
        progress.empty()

    return best_params, best_res

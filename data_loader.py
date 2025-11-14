import yfinance as yf
import pandas as pd

def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    return df

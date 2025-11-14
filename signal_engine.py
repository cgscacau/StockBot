import pandas as pd
from strategies import StrategyConfig
from indicators import ema, rsi, atr

def generate_signals(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = df.copy()
    df['EMA_fast'] = ema(df['Close'], cfg.fast_ema)
    df['EMA_slow'] = ema(df['Close'], cfg.slow_ema)
    df['RSI'] = rsi(df['Close'], cfg.rsi_length)
    df['ATR'] = atr(df, cfg.atr_length)

    df['signal'] = 0  # 1 = compra, -1 = venda, 0 = neutro

    # Compra quando fast cruza acima de slow e RSI sobe acima do nível de compra
    buy_condition = (
        (df['EMA_fast'] > df['EMA_slow']) &
        (df['EMA_fast'].shift() <= df['EMA_slow'].shift()) &
        (df['RSI'] > cfg.rsi_buy)
    )

    # Venda quando fast cruza abaixo de slow e RSI cai abaixo do nível de venda
    sell_condition = (
        (df['EMA_fast'] < df['EMA_slow']) &
        (df['EMA_fast'].shift() >= df['EMA_slow'].shift()) &
        (df['RSI'] < cfg.rsi_sell)
    )

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    # Pontos de entrada / SL / TP
    df['entry'] = None
    df['stop_loss'] = None
    df['take_profit'] = None
    df['direction'] = None

    for idx in df.index:
        sig = df.at[idx, 'signal']
        if sig == 1:
            price = df.at[idx, 'Close']
            atr_val = df.at[idx, 'ATR']
            sl = price - atr_val
            tp = price + cfg.risk_reward * (price - sl)
            df.at[idx, 'entry'] = price
            df.at[idx, 'stop_loss'] = sl
            df.at[idx, 'take_profit'] = tp
            df.at[idx, 'direction'] = 'BUY'
        elif sig == -1:
            price = df.at[idx, 'Close']
            atr_val = df.at[idx, 'ATR']
            sl = price + atr_val
            tp = price - cfg.risk_reward * (sl - price)
            df.at[idx, 'entry'] = price
            df.at[idx, 'stop_loss'] = sl
            df.at[idx, 'take_profit'] = tp
            df.at[idx, 'direction'] = 'SELL'

    return df

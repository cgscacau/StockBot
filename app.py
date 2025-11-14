import streamlit as st
import pandas as pd

from data_loader import get_history
from strategies import StrategyConfig
from signal_engine import generate_signals
from backtester import backtest
from optimizer import optimize_parameters

st.set_page_config(page_title="Scanner de Oportunidades", layout="wide")

st.title("📈 Scanner de Oportunidades – Price Action & TA")

# --- SIDEBAR ---
tickers_input = st.sidebar.text_input(
    "Tickers (separados por vírgula)",
    value="PETR4.SA, VALE3.SA"
)

period = st.sidebar.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1h", "15m"], index=0)

# Estratégia padrão
base_cfg = StrategyConfig(
    name="EMA+RSI",
    fast_ema=9,
    slow_ema=21,
    rsi_length=14,
    rsi_buy=30,
    rsi_sell=70,
    atr_length=14,
    risk_reward=2.0,
    risk_pct=0.02
)

if st.sidebar.button("📊 Rodar Scanner"):
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    results = []

    for t in tickers:
        df = get_history(t, period=period, interval=interval)
        df_sig = generate_signals(df, base_cfg)
        last_signal = df_sig[df_sig['signal'] != 0].tail(1)

        if not last_signal.empty:
            row = last_signal.iloc[0]
            results.append({
                "Ticker": t,
                "Data": row.name,
                "Direção": row['direction'],
                "Entrada": row['entry'],
                "Stop Loss": row['stop_loss'],
                "Stop Gain": row['take_profit']
            })

    if results:
        st.subheader("Sinais recentes encontrados")
        st.dataframe(pd.DataFrame(results))
    else:
        st.info("Nenhum sinal recente encontrado para os parâmetros atuais.")

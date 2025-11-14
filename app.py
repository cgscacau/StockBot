import streamlit as st
import pandas as pd

from modules.data_loader import get_history
from modules.engine_rules import generate_rule_signals, backtest, optimize
from modules.engine_ml import predict, ml_to_trade


st.set_page_config(page_title="StockBot Modular", layout="wide")
st.title("📊 StockBot Modular – Scanner, Backtest e ML")

# ----------------- SIDEBAR -----------------
tickers = st.sidebar.text_input("Tickers", "PETR4.SA, VALE3.SA")
period = st.sidebar.selectbox("Período", ["6mo", "1y", "2y", "5y"])
interval = st.sidebar.selectbox("Intervalo", ["1d", "1h"])

rr = st.sidebar.slider("RR (Take Profit = RR x risco)", 1.0, 4.0, 2.0)
atr_mult = st.sidebar.slider("Multiplicador ATR (Stop)", 0.5, 3.0, 1.0)

lookback = st.sidebar.slider("Lookback ML", 5, 20, 10)

engine = st.sidebar.selectbox(
    "Motor de Sinais",
    ["Tradicional (EMAs)", "Machine Learning", "Híbrido"]
)

tickers = [t.strip() for t in tickers.split(",") if t.strip()]


# ----------------- TABS -----------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Scanner",
    "📈 Tradicional (Backtest / Otimização)",
    "🤖 Machine Learning"
])


# ========================= TAB 1 – SCANNER =========================
with tab1:
    if st.button("Rodar Scanner"):
        results = []

        for t in tickers:
            df = get_history(t, period, interval)
            if df.empty:
                continue

            row = {"Ticker": t}

            # Motor Tradicional
            df_rule = generate_rule_signals(df)
            last = df_rule[df_rule["signal"] != 0].tail(1)

            if not last.empty:
                r = last.iloc[0]
                row.update({
                    "rule_dir": r["direction"],
                    "rule_entry": r["entry"],
                    "rule_sl": r["stop_loss"],
                    "rule_tp": r["take_profit"]
                })

            # Machine Learning
            if engine in ["Machine Learning", "Híbrido"]:
                prob, last_row = predict(df, lookback)
                if prob is not None:
                    trade = ml_to_trade(prob, last_row, rr=rr, atr_mult=atr_mult)
                    row.update({
                        "ml_prob": prob,
                        "ml_dir": trade["direction"],
                        "ml_entry": trade["entry"],
                        "ml_sl": trade["stop"],
                        "ml_tp": trade["tp"]
                    })

            results.append(row)

        st.dataframe(pd.DataFrame(results))


# ========================= TAB 2 – MOTOR TRADICIONAL =========================
with tab2:
    ticker_bt = st.selectbox("Ticker Backtest", tickers)
    if st.button("Backtest Tradicional"):
        df = get_history(ticker_bt, period, interval)
        df_sig = generate_rule_signals(df, rr=rr, atr_mult=atr_mult)
        res = backtest(df_sig)
        st.write(f"Retorno total: {res['return']*100:.2f}%")
        st.write(f"Max DD: {res['max_dd']*100:.2f}%")
        st.line_chart(res["equity"])

    st.markdown("---")
    if st.button("Otimizar Parâmetros"):
        df = get_history(ticker_bt, period, interval)
        progress = st.progress(0)
        best, result = optimize(
            df,
            fast_list=range(5, 15),
            slow_list=range(15, 40, 2),
            rr_list=[1.5, 2.0, 2.5],
            atr_list=[0.5, 1.0, 1.5],
            progress=progress
        )
        st.success(f"Melhores parâmetros: {best}")
        st.write(result)


# ========================= TAB 3 – MACHINE LEARNING =========================
with tab3:
    ticker_ml = st.selectbox("Ticker ML", tickers)
    if st.button("Rodar ML"):
        df = get_history(ticker_ml, period, interval)
        prob, last_row = predict(df, lookback)
        trade = ml_to_trade(prob, last_row, rr=rr, atr_mult=atr_mult)
        st.write("Probabilidade de alta:", prob)
        st.write(trade)

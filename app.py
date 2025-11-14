import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import ta

# ========================= CONFIG STREAMLIT =========================
st.set_page_config(
    page_title="Scanner de Oportunidades – ML & Análise Técnica",
    layout="wide"
)

st.title("📊 Scanner de Oportunidades com Machine Learning & Análise Técnica")

# ========================= FUNÇÕES AUXILIARES =========================

def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Cálculo manual do ATR para evitar bug da biblioteca ta.
    """
    if df.empty:
        return pd.Series(dtype=float)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr


# ========================= FUNÇÕES DE DADOS =========================

@st.cache_data(show_spinner=False)
def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.dropna(inplace=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona indicadores técnicos usados no ML (MACD, DMI, KST, etc.)."""
    df = df.copy()
    if df.empty:
        return df

    # MACD
    macd_obj = ta.trend.MACD(df["Close"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

    # ADX / DMI
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["adx"] = adx.adx()
    df["+di"] = adx.adx_pos()
    df["-di"] = adx.adx_neg()

    # KST
    kst = ta.trend.KSTIndicator(df["Close"])
    df["kst"] = kst.kst()
    df["kst_signal"] = kst.kst_sig()

    # ATR (manual – sem usar ta.volatility)
    df["atr"] = compute_atr(df)

    # EMAs (para motor tradicional também)
    df["ema_fast"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=21, adjust=False).mean()

    df.dropna(inplace=True)
    return df


# ========================= MOTOR TRADICIONAL =========================

def generate_rule_signals(df: pd.DataFrame,
                          fast_ema: int = 9,
                          slow_ema: int = 21,
                          rr: float = 2.0,
                          atr_mult: float = 1.0) -> pd.DataFrame:
    """Sinais baseados em cruzamento de EMAs + ATR p/ SL/TP."""
    df = df.copy()
    if df.empty:
        # garante colunas esperadas, mesmo vazio
        for col in ["ema_fast", "ema_slow", "atr", "signal", "direction",
                    "entry", "stop_loss", "take_profit"]:
            df[col] = np.nan
        return df

    df["ema_fast"] = df["Close"].ewm(span=fast_ema, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=slow_ema, adjust=False).mean()
    df["atr"] = compute_atr(df)

    df["signal"] = 0
    buy = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    sell = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))

    df.loc[buy, "signal"] = 1
    df.loc[sell, "signal"] = -1

    df["direction"] = None
    df["entry"] = np.nan
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan

    for idx in df.index:
        sig = df.at[idx, "signal"]
        if sig == 1:
            price = df.at[idx, "Close"]
            atr_val = df.at[idx, "atr"]
            if pd.isna(atr_val):
                continue
            sl = price - atr_mult * atr_val
            tp = price + rr * (price - sl)
            df.at[idx, "direction"] = "BUY"
            df.at[idx, "entry"] = price
            df.at[idx, "stop_loss"] = sl
            df.at[idx, "take_profit"] = tp
        elif sig == -1:
            price = df.at[idx, "Close"]
            atr_val = df.at[idx, "atr"]
            if pd.isna(atr_val):
                continue
            sl = price + atr_mult * atr_val
            tp = price - rr * (sl - price)
            df.at[idx, "direction"] = "SELL"
            df.at[idx, "entry"] = price
            df.at[idx, "stop_loss"] = sl
            df.at[idx, "take_profit"] = tp

    return df


def backtest_rule(df: pd.DataFrame,
                  initial_capital: float = 100_000.0) -> dict:
    """
    Backtest simples: entra 100% do capital no sinal, zera no próximo sinal oposto.
    Ignora SL/TP intradiário (mantém até próximo sinal).
    """
    df = df.copy()
    if df.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "equity_curve": []}

    df["ret"] = df["Close"].pct_change().fillna(0)

    position = 0  # 1 comprado, -1 vendido
    equity = initial_capital
    equity_curve = []

    for i in range(1, len(df)):
        sig = df["signal"].iloc[i]
        price_ret = df["ret"].iloc[i]

        # Atualiza equity com posição atual
        equity *= (1 + position * price_ret)

        # Atualiza posição se houver novo sinal
        if sig == 1:
            position = 1
        elif sig == -1:
            position = -1

        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)
    if len(equity_curve):
        total_return = (equity_curve[-1] / initial_capital) - 1
        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_dd = drawdown.min()
    else:
        total_return = 0.0
        max_dd = 0.0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "equity_curve": equity_curve
    }


def optimize_rule(df: pd.DataFrame,
                  ema_fast_list,
                  ema_slow_list,
                  rr_list,
                  atr_list):
    """Grid search simples sobre os parâmetros do motor tradicional."""
    if df.empty:
        return None, None, None

    best_metric = -1e9
    best_params = None
    best_result = None

    for f, s, rr, am in product(ema_fast_list, ema_slow_list, rr_list, atr_list):
        if f >= s:
            continue  # ignora combinações estranhas
        sig_df = generate_rule_signals(df, fast_ema=f, slow_ema=s, rr=rr, atr_mult=am)
        res = backtest_rule(sig_df)
        # métrica simples: retorno - penalização por drawdown
        metric = res["total_return"] - abs(res["max_drawdown"])

        if metric > best_metric:
            best_metric = metric
            best_params = (f, s, rr, am)
            best_result = res

    return best_params, best_result, best_metric


# ========================= MOTOR MACHINE LEARNING =========================

def prepare_ml_dataset(df: pd.DataFrame, lookback: int = 10):
    """Prepara dados com indicadores e janelas temporais."""
    df = add_indicators(df)
    if df.empty:
        return df, [], None, None, None, None, None

    # Target: 1 se fechar amanhã acima de hoje
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    if df.empty:
        return df, [], None, None, None, None, None

    feature_cols = [c for c in df.columns if c not in ["target"]]
    X_full = df[feature_cols].values
    y_full = df["target"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_full)

    # Janelas para LSTM/GRU
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - lookback):
        X_seq.append(X_scaled[i:i+lookback])
        y_seq.append(y_full[i+lookback])

    if not X_seq:
        return df, feature_cols, scaler, X_scaled, y_full, None, None

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return df, feature_cols, scaler, X_scaled, y_full, X_seq, y_seq


def train_random_forest(X_scaled, y):
    """Ensemble clássico (RandomForest)."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_scaled, y)
    return model


def train_lstm_gru(X_seq, y_seq, model_type="GRU"):
    """Treina um modelo LSTM ou GRU simples."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    except Exception:
        st.warning("TensorFlow não disponível. Usando apenas RandomForest.")
        return None

    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=X_seq.shape[1:]))
    else:
        model.add(GRU(64, return_sequences=True, input_shape=X_seq.shape[1:]))

    model.add(Dropout(0.2))
    if model_type == "LSTM":
        model.add(LSTM(32))
    else:
        model.add(GRU(32))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X_seq, y_seq, epochs=8, batch_size=16, verbose=0)

    return model


def ml_predict_last(df_raw: pd.DataFrame,
                    lookback: int = 10,
                    use_lstm: bool = True,
                    use_gru: bool = True,
                    use_rf: bool = True):
    """
    Treina modelos e retorna probabilidade de alta no último ponto.
    Faz ensemble GRU > LSTM > RF, como no card.
    """
    df, feature_cols, scaler, X_scaled, y_full, X_seq, y_seq = prepare_ml_dataset(df_raw, lookback)

    if X_seq is None or X_scaled is None or X_scaled.size == 0:
        return None, None

    # separa último ponto para previsão
    X_last_seq = X_seq[-1].reshape(1, *X_seq.shape[1:])
    X_last_flat = X_scaled[-1].reshape(1, -1)

    preds = []
    pesos = []

    # GRU
    if use_gru and y_seq is not None:
        gru_model = train_lstm_gru(X_seq, y_seq, model_type="GRU")
        if gru_model is not None:
            p_gru = float(gru_model.predict(X_last_seq)[0][0])
            preds.append(p_gru)
            pesos.append(0.5)   # maior peso (card diz GRU > LSTM)

    # LSTM
    if use_lstm and y_seq is not None:
        lstm_model = train_lstm_gru(X_seq, y_seq, model_type="LSTM")
        if lstm_model is not None:
            p_lstm = float(lstm_model.predict(X_last_seq)[0][0])
            preds.append(p_lstm)
            pesos.append(0.3)

    # RandomForest (ensemble clássico)
    if use_rf:
        rf_model = train_random_forest(X_scaled, y_full)
        p_rf = float(rf_model.predict_proba(X_last_flat)[0][1])
        preds.append(p_rf)
        pesos.append(0.2)

    if not preds:
        return None, None

    preds = np.array(preds)
    pesos = np.array(pesos)
    pesos = pesos / pesos.sum()

    final_prob = float((preds * pesos).sum())
    last_row = df.iloc[-1]

    return final_prob, last_row


def ml_signal_to_trade(prob_up: float,
                       last_row: pd.Series,
                       rr: float = 2.0,
                       atr_mult: float = 1.0,
                       buy_th: float = 0.6,
                       sell_th: float = 0.4):
    """Transforma probabilidade em direção/entrada/SL/TP."""
    if prob_up is None or last_row is None:
        return None

    direction = "NEUTRAL"
    if prob_up >= buy_th:
        direction = "BUY"
    elif prob_up <= sell_th:
        direction = "SELL"

    price = last_row["Close"]
    atr_val = last_row.get("atr", np.nan)
    if pd.isna(atr_val):
        atr_val = compute_atr(last_row.to_frame().T).iloc[-1]

    if direction == "BUY":
        sl = price - atr_mult * atr_val
        tp = price + rr * (price - sl)
    elif direction == "SELL":
        sl = price + atr_mult * atr_val
        tp = price - rr * (sl - price)
    else:
        sl = tp = None

    return {
        "direction": direction,
        "entry": float(price),
        "stop_loss": float(sl) if sl is not None else None,
        "take_profit": float(tp) if tp is not None else None
    }


# ========================= INTERFACE STREAMLIT =========================

# ----- SIDEBAR -----
st.sidebar.header("Configurações")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por vírgula)",
    value="PETR4.SA, VALE3.SA"
)

period = st.sidebar.selectbox("Período", ["6mo", "1y", "2y", "5y"], index=1)
interval = st.sidebar.selectbox("Intervalo", ["1d", "1h"], index=0)

engine = st.sidebar.radio(
    "Motor de Sinais",
    ["Tradicional (EMAs)", "Machine Learning", "Híbrido"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros Gerais")

rr = st.sidebar.slider("Risco/Retorno (TP = RR x risco)", 1.0, 4.0, 2.0, 0.5)
atr_mult = st.sidebar.slider("Multiplicador ATR para Stop Loss", 0.5, 3.0, 1.0, 0.5)

st.sidebar.markdown("---")
if engine != "Tradicional (EMAs)":
    lookback = st.sidebar.slider("Lookback ML (dias)", 5, 20, 10)

# ----- TABS -----
tab1, tab2, tab3 = st.tabs([
    "🔍 Scanner de Oportunidades",
    "📈 Detalhe & Backtest (Motor Tradicional)",
    "🤖 Detalhe Machine Learning"
])

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# ========================= TAB 1: SCANNER =========================
with tab1:
    st.subheader("🔍 Scanner de Oportunidades")

    if st.button("Rodar Scanner", type="primary"):
        rows = []

        for t in tickers:
            df_raw = get_history(t, period=period, interval=interval)
            if df_raw.empty:
                continue

            # Motor tradicional
            rule_row = {}
            df_rule = generate_rule_signals(df_raw, rr=rr, atr_mult=atr_mult)
            last_sig = df_rule[df_rule["signal"] != 0].tail(1)
            if not last_sig.empty:
                r = last_sig.iloc[0]
                rule_row = {
                    "rule_direction": r["direction"],
                    "rule_entry": r["entry"],
                    "rule_sl": r["stop_loss"],
                    "rule_tp": r["take_profit"],
                    "rule_date": r.name
                }

            # Motor ML
            ml_row = {}
            if engine != "Tradicional (EMAs)":
                prob_up, last_row = ml_predict_last(df_raw, lookback=lookback)
                if prob_up is not None and last_row is not None:
                    trade = ml_signal_to_trade(prob_up, last_row, rr=rr, atr_mult=atr_mult)
                    if trade is not None:
                        ml_row = {
                            "ml_prob_up": prob_up,
                            "ml_direction": trade["direction"],
                            "ml_entry": trade["entry"],
                            "ml_sl": trade["stop_loss"],
                            "ml_tp": trade["take_profit"]
                        }

            combined = {"Ticker": t}
            combined.update(rule_row)
            combined.update(ml_row)
            rows.append(combined)

        if rows:
            df_res = pd.DataFrame(rows)
            if "ml_prob_up" in df_res.columns:
                df_res["ml_prob_up"] = (df_res["ml_prob_up"] * 100).round(2)
            st.dataframe(df_res)
        else:
            st.info("Nenhum dado encontrado.")


# ========================= TAB 2: BACKTEST MOTOR TRADICIONAL =========================
with tab2:
    st.subheader("📈 Backtest & Otimização – Motor Tradicional")

    ticker_bt = st.selectbox("Ticker para backtest", tickers) if tickers else None

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        ema_fast = st.slider("EMA Rápida", 3, 30, 9)
        ema_slow = st.slider("EMA Lenta", 10, 60, 21)
    with col_bt2:
        do_opt = st.checkbox("Otimizar parâmetros", value=False)

    if ticker_bt and st.button("Rodar Backtest Tradicional"):
        df_raw = get_history(ticker_bt, period=period, interval=interval)
        if df_raw.empty:
            st.warning("Sem dados para este ticker/período.")
        else:
            if do_opt:
                with st.spinner("Otimizando parâmetros..."):
                    params, res, metric = optimize_rule(
                        df_raw,
                        ema_fast_list=range(5, 15),
                        ema_slow_list=range(15, 40, 2),
                        rr_list=[1.5, 2.0, 2.5],
                        atr_list=[0.5, 1.0, 1.5]
                    )
                if params is None:
                    st.warning("Não foi possível otimizar parâmetros.")
                else:
                    f, s, rr_best, am_best = params
                    st.success(f"Melhor combinação: EMA_f={f}, EMA_s={s}, RR={rr_best}, ATR_mult={am_best:.1f}")
                    st.write(f"Retorno total: {res['total_return']*100:.2f}%")
                    st.write(f"Max Drawdown: {res['max_drawdown']*100:.2f}%")

                    if res and len(res["equity_curve"]):
                        eq = pd.Series(res["equity_curve"])
                        st.line_chart(eq)
            else:
                df_sig = generate_rule_signals(df_raw, fast_ema=ema_fast,
                                               slow_ema=ema_slow,
                                               rr=rr,
                                               atr_mult=atr_mult)
                res = backtest_rule(df_sig)
                st.write(f"Retorno total: {res['total_return']*100:.2f}%")
                st.write(f"Max Drawdown: {res['max_drawdown']*100:.2f}%")
                if res and len(res["equity_curve"]):
                    eq = pd.Series(res["equity_curve"])
                    st.line_chart(eq)


# ========================= TAB 3: DETALHE MACHINE LEARNING =========================
with tab3:
    st.subheader("🤖 Detalhe Machine Learning")

    ticker_ml = st.selectbox("Ticker p/ análise ML", tickers, key="ticker_ml") if tickers else None

    if ticker_ml and st.button("Rodar ML para este ticker"):
        df_raw = get_history(ticker_ml, period=period, interval=interval)
        if df_raw.empty:
            st.warning("Sem dados para este ticker/período.")
        else:
            prob_up, last_row = ml_predict_last(df_raw, lookback=lookback)

            if prob_up is None or last_row is None:
                st.warning("Poucos dados ou erro ao treinar modelos.")
            else:
                trade = ml_signal_to_trade(prob_up, last_row, rr=rr, atr_mult=atr_mult)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probabilidade de Alta (ensemble GRU/LSTM/RF)",
                              f"{prob_up*100:.2f}%")
                with col2:
                    st.metric("Direção sugerida", trade["direction"])

                st.markdown("### Níveis sugeridos")
                st.write(f"Entrada: **{trade['entry']:.2f}**")
                if trade["stop_loss"] is not None:
                    st.write(f"Stop Loss: **{trade['stop_loss']:.2f}**")
                    st.write(f"Take Profit: **{trade['take_profit']:.2f}**")

                st.markdown("### Últimos candles com indicadores")
                df_ind = add_indicators(df_raw).tail(60)
                st.dataframe(df_ind[["Close", "macd", "macd_signal",
                                     "rsi", "adx", "kst", "kst_signal"]])

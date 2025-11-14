import os
import json
import hashlib

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

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

# ========================= CONFIG CACHE JSON (OTIMIZAÇÃO) =========================

CACHE_DIR = "opt_cache"


def _ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def make_opt_key(ticker, period, interval,
                 ema_fast_list, ema_slow_list, rr_list, atr_list):
    """
    Gera uma chave única para o cache de otimização,
    baseada em ticker, período, intervalo e grid de parâmetros.
    """
    payload = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "fast": list(ema_fast_list),
        "slow": list(ema_slow_list),
        "rr": list(rr_list),
        "atr": list(atr_list),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_opt_from_json(key):
    """
    Tenta carregar parâmetros otimizados e equity curve a partir de JSON.
    """
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"opt_{key}.json")
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)

        params = tuple(data["params"])
        res = {
            "total_return": float(data["total_return"]),
            "max_drawdown": float(data["max_drawdown"]),
            "equity_curve": np.array(data["equity_curve"], dtype=float),
        }
        return params, res
    except Exception:
        return None


def save_opt_to_json(key, params, res):
    """
    Salva resultado da otimização em JSON para reuso.
    """
    _ensure_cache_dir()
    path = os.path.join(CACHE_DIR, f"opt_{key}.json")

    payload = {
        "params": list(params),
        "total_return": float(res["total_return"]),
        "max_drawdown": float(res["max_drawdown"]),
        "equity_curve": [float(x) for x in res["equity_curve"]],
    }
    with open(path, "w") as f:
        json.dump(payload, f)


# ========================= FUNÇÕES AUXILIARES =========================

def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Cálculo manual do ATR para evitar bugs da lib ta.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=1).mean()
    return atr


# ========================= DADOS & INDICADORES (COM CACHE) =========================

@st.cache_data(show_spinner=False)
def get_history(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Download de histórico via yfinance (com cache).
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.dropna(inplace=True)
    return df


@st.cache_data(show_spinner=False)
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona indicadores técnicos com try/except para evitar
    que erros na lib `ta` derrubem o app.
    """
    df = df.copy()
    if df.empty:
        return df

    # MACD
    try:
        macd_obj = ta.trend.MACD(df["Close"])
        df["macd"] = macd_obj.macd()
        df["macd_signal"] = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()
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

    # ATR manual
    df["atr"] = compute_atr(df)

    df.dropna(inplace=True)
    return df


# ========================= MOTOR TRADICIONAL (REGRAS) =========================

def generate_rule_signals(df: pd.DataFrame,
                          fast_ema: int = 9,
                          slow_ema: int = 21,
                          rr: float = 2.0,
                          atr_mult: float = 1.0) -> pd.DataFrame:
    """
    Sinais baseados em cruzamento de EMAs + ATR para SL/TP (vetorizado).
    """
    df = df.copy()
    if df.empty:
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

    df["direction"] = "NEUTRAL"
    df["direction"] = df["direction"].astype(object)
    df.loc[df["signal"] == 1, "direction"] = "BUY"
    df.loc[df["signal"] == -1, "direction"] = "SELL"

    df["entry"] = df["Close"].where(df["signal"] != 0, np.nan)
    df["stop_loss"] = np.nan
    df["take_profit"] = np.nan

    # BUY
    buy_rows = df["signal"] == 1
    df.loc[buy_rows, "stop_loss"] = df["Close"] - atr_mult * df["atr"]
    df.loc[buy_rows, "take_profit"] = df["Close"] + rr * (df["Close"] - df["stop_loss"])

    # SELL
    sell_rows = df["signal"] == -1
    df.loc[sell_rows, "stop_loss"] = df["Close"] + atr_mult * df["atr"]
    df.loc[sell_rows, "take_profit"] = df["Close"] - rr * (df["stop_loss"] - df["Close"])

    return df


def backtest_rule(df: pd.DataFrame,
                  initial_capital: float = 100_000.0) -> dict:
    """
    Backtest simples: entra no sinal atual (1 ou -1) e mantém até próximo sinal.
    """
    df = df.copy()
    if df.empty:
        return {"total_return": 0.0, "max_drawdown": 0.0, "equity_curve": np.array([])}

    df["ret"] = df["Close"].pct_change().fillna(0)

    position = 0
    equity = initial_capital
    equity_curve = []

    for i in range(1, len(df)):
        sig = df["signal"].iloc[i]
        price_ret = df["ret"].iloc[i]

        equity *= (1 + position * price_ret)

        if sig == 1:
            position = 1
        elif sig == -1:
            position = -1

        equity_curve.append(equity)

    equity_curve = np.array(equity_curve, dtype=float)
    if len(equity_curve) == 0:
        return {"total_return": 0.0, "max_drawdown": 0.0, "equity_curve": np.array([])}

    total_return = (equity_curve[-1] / initial_capital) - 1
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min() if len(drawdown) else 0.0

    return {
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "equity_curve": equity_curve
    }


def optimize_rule(df: pd.DataFrame,
                  ema_fast_list,
                  ema_slow_list,
                  rr_list,
                  atr_list,
                  progress_bar=None):
    """
    Grid search do motor tradicional (EMAs + RR + ATR).
    Pode usar progress_bar do Streamlit.
    """
    if df.empty:
        return None, None, None

    combos = [
        (f, s, rr, am)
        for f in ema_fast_list
        for s in ema_slow_list
        for rr in rr_list
        for am in atr_list
        if f < s
    ]
    total = len(combos)
    if total == 0:
        return None, None, None

    best_metric = -1e9
    best_params = None
    best_result = None

    for i, (f, s, rr, am) in enumerate(combos, start=1):
        sig_df = generate_rule_signals(df, fast_ema=f, slow_ema=s, rr=rr, atr_mult=am)
        res = backtest_rule(sig_df)
        metric = res["total_return"] - abs(res["max_drawdown"])

        if metric > best_metric:
            best_metric = metric
            best_params = (f, s, rr, am)
            best_result = res

        if progress_bar is not None:
            progress_bar.progress(i / total)

    if progress_bar is not None:
        progress_bar.empty()

    return best_params, best_result, best_metric


# ========================= MOTOR MACHINE LEARNING =========================

def prepare_ml_dataset(df: pd.DataFrame, lookback: int = 10):
    """
    Prepara dataset para ML (indicadores + janelas temporais).
    """
    df = df.copy()
    if df.empty or len(df) < 80:
        return df, [], None, None, None, None, None

    df = add_indicators(df)
    if df.empty:
        return df, [], None, None, None, None, None

    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    if df.empty:
        return df, [], None, None, None, None, None

    feature_cols = [c for c in df.columns if c not in ["target"]]
    X_full = df[feature_cols].values
    y_full = df["target"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_full)

    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - lookback):
        X_seq.append(X_scaled[i:i+lookback])
        y_seq.append(y_full[i+lookback])

    if not X_seq:
        return df, feature_cols, scaler, X_scaled, y_full, None, None

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    return df, feature_cols, scaler, X_scaled, y_full, X_seq, y_seq


@st.cache_resource
def train_random_forest_cached(X_scaled, y):
    """
    Treina RandomForest com cache de recurso (não recalcula a cada clique).
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(X_scaled, y)
    return model


def train_lstm_gru(X_seq, y_seq, model_type="GRU"):
    """
    Treina modelo LSTM ou GRU (sem cache, pode ser mais pesado).
    """
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
    Ensemble GRU > LSTM > RF.
    """
    df, feature_cols, scaler, X_scaled, y_full, X_seq, y_seq = prepare_ml_dataset(df_raw, lookback)

    if X_seq is None or X_scaled is None or X_scaled.size == 0:
        return None, None

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
            pesos.append(0.5)

    # LSTM
    if use_lstm and y_seq is not None:
        lstm_model = train_lstm_gru(X_seq, y_seq, model_type="LSTM")
        if lstm_model is not None:
            p_lstm = float(lstm_model.predict(X_last_seq)[0][0])
            preds.append(p_lstm)
            pesos.append(0.3)

    # RandomForest
    if use_rf:
        rf_model = train_random_forest_cached(X_scaled, y_full)
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
    """
    Converte probabilidade de alta em direção/entrada/SL/TP.
    """
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
else:
    lookback = 10  # valor padrão qualquer

tab1, tab2, tab3 = st.tabs([
    "🔍 Scanner de Oportunidades",
    "📈 Backtest & Otimização (Tradicional)",
    "🤖 Detalhe Machine Learning"
])

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# ========================= TAB 1 – SCANNER =========================
with tab1:
    st.subheader("🔍 Scanner de Oportunidades")

    if st.button("Rodar Scanner", type="primary"):
        rows = []

        for t in tickers:
            df_raw = get_history(t, period=period, interval=interval)
            if df_raw.empty:
                continue

            row = {"Ticker": t}

            # Motor Tradicional
            df_rule = generate_rule_signals(df_raw, rr=rr, atr_mult=atr_mult)
            last_sig = df_rule[df_rule["signal"] != 0].tail(1)
            if not last_sig.empty:
                r = last_sig.iloc[0]
                row.update({
                    "rule_direction": r["direction"],
                    "rule_entry": r["entry"],
                    "rule_sl": r["stop_loss"],
                    "rule_tp": r["take_profit"],
                    "rule_date": r.name
                })

            # Motor ML
            if engine != "Tradicional (EMAs)":
                prob_up, last_row = ml_predict_last(df_raw, lookback=lookback)
                if prob_up is not None and last_row is not None:
                    trade = ml_signal_to_trade(prob_up, last_row, rr=rr, atr_mult=atr_mult)
                    if trade is not None:
                        row.update({
                            "ml_prob_up": prob_up,
                            "ml_direction": trade["direction"],
                            "ml_entry": trade["entry"],
                            "ml_sl": trade["stop_loss"],
                            "ml_tp": trade["take_profit"]
                        })

            rows.append(row)

        if rows:
            df_res = pd.DataFrame(rows)
            if "ml_prob_up" in df_res.columns:
                df_res["ml_prob_up"] = (df_res["ml_prob_up"] * 100).round(2)
            st.dataframe(df_res)
        else:
            st.info("Nenhum dado encontrado.")


# ========================= TAB 2 – BACKTEST & OTIMIZAÇÃO =========================
with tab2:
    st.subheader("📈 Backtest & Otimização – Motor Tradicional (EMAs)")

    ticker_bt = st.selectbox("Ticker para backtest", tickers) if tickers else None

    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        ema_fast = st.slider("EMA Rápida", 3, 30, 9)
        ema_slow = st.slider("EMA Lenta", 10, 60, 21)
    with col_bt2:
        do_opt = st.checkbox("Otimizar parâmetros (EMAs, RR, ATR)", value=False)

    if ticker_bt and st.button("Rodar Backtest Tradicional"):
        df_raw = get_history(ticker_bt, period=period, interval=interval)
        if df_raw.empty:
            st.warning("Sem dados para este ticker/período.")
        else:
            if do_opt:
                # define grid da otimização (mesmo comportamento, mas agora com cache JSON)
                ema_fast_list = list(range(5, 15))
                ema_slow_list = list(range(15, 40, 2))
                rr_list = [1.5, 2.0, 2.5]
                atr_list = [0.5, 1.0, 1.5]

                key = make_opt_key(
                    ticker_bt, period, interval,
                    ema_fast_list, ema_slow_list, rr_list, atr_list
                )

                # Tenta carregar do JSON
                cached = load_opt_from_json(key)
                if cached is not None:
                    params, res = cached
                    st.info("✅ Otimização carregada do cache (JSON).")
                else:
                    st.write("⏱ Otimizando parâmetros, isso pode levar alguns segundos...")
                    progress = st.progress(0.0)
                    params, res, metric = optimize_rule(
                        df_raw,
                        ema_fast_list=ema_fast_list,
                        ema_slow_list=ema_slow_list,
                        rr_list=rr_list,
                        atr_list=atr_list,
                        progress_bar=progress
                    )
                    if params is not None and res is not None:
                        save_opt_to_json(key, params, res)
                        st.success("✅ Otimização concluída e salva em cache (JSON).")

                if params is None:
                    st.warning("Não foi possível otimizar parâmetros.")
                else:
                    f, s, rr_best, am_best = params
                    st.write(f"Melhor combinação: EMA_f={f}, EMA_s={s}, RR={rr_best}, ATR_mult={am_best:.1f}")
                    st.write(f"Retorno total: {res['total_return']*100:.2f}%")
                    st.write(f"Max Drawdown: {res['max_drawdown']*100:.2f}%")
                    if res and len(res["equity_curve"]):
                        st.line_chart(pd.Series(res["equity_curve"]))
            else:
                df_sig = generate_rule_signals(df_raw, fast_ema=ema_fast,
                                               slow_ema=ema_slow,
                                               rr=rr,
                                               atr_mult=atr_mult)
                res = backtest_rule(df_sig)
                st.write(f"Retorno total: {res['total_return']*100:.2f}%")
                st.write(f"Max Drawdown: {res['max_drawdown']*100:.2f}%")
                if res and len(res["equity_curve"]):
                    st.line_chart(pd.Series(res["equity_curve"]))


# ========================= TAB 3 – DETALHE ML =========================
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
                st.warning("Poucos dados ou erro ao treinar modelos (tente aumentar o período).")
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

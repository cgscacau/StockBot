import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from .data_loader import add_indicators, compute_atr


# ========================= PREPARAÇÃO =========================
def prepare(df, lookback=10):
    if df.empty or len(df) < 80:
        return None

    df = add_indicators(df)
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)

    feats = [c for c in df.columns if c not in ["target"]]
    X = df[feats].values
    y = df["target"].values

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    Xseq, yseq = [], []
    for i in range(len(Xs) - lookback):
        Xseq.append(Xs[i:i+lookback])
        yseq.append(y[i+lookback])

    return np.array(Xseq), np.array(yseq), Xs, y, df.iloc[-1]


# ========================= RANDOM FOREST =========================
def train_rf(Xs, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )
    model.fit(Xs, y)
    return model


# ========================= LSTM / GRU =========================
def train_rnn(Xseq, yseq, mode="GRU"):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
    except Exception:
        return None

    model = Sequential()
    if mode == "LSTM":
        model.add(LSTM(64, return_sequences=True, input_shape=Xseq.shape[1:]))
    else:
        model.add(GRU(64, return_sequences=True, input_shape=Xseq.shape[1:]))

    model.add(Dropout(0.2))
    if mode == "LSTM":
        model.add(LSTM(32))
    else:
        model.add(GRU(32))

    model.add(Dense(16, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(Xseq, yseq, epochs=8, batch_size=16, verbose=0)

    return model


# ========================= PREVISÃO FINAL =========================
def predict(df_raw, lookback=10):
    prep = prepare(df_raw, lookback)
    if prep is None:
        return None, None

    Xseq, yseq, Xs, y, last_row = prep

    X_last_seq = Xseq[-1].reshape(1, *Xseq.shape[1:])
    X_last_flat = Xs[-1].reshape(1, -1)

    preds = []
    weights = []

    # GRU
    gru = train_rnn(Xseq, yseq, mode="GRU")
    if gru:
        preds.append(float(gru.predict(X_last_seq)[0][0]))
        weights.append(0.5)

    # LSTM
    lstm = train_rnn(Xseq, yseq, mode="LSTM")
    if lstm:
        preds.append(float(lstm.predict(X_last_seq)[0][0]))
        weights.append(0.3)

    # Random Forest
    rf = train_rf(Xs, y)
    preds.append(float(rf.predict_proba(X_last_flat)[0][1]))
    weights.append(0.2)

    w = np.array(weights) / sum(weights)
    final_prob = float((np.array(preds) * w).sum())

    return final_prob, last_row


# ========================= TRANSFORMA PREVISÃO EM TRADE =========================
def ml_to_trade(prob, row, rr=2.0, atr_mult=1.0, buy_th=0.6, sell_th=0.4):
    price = row["Close"]
    atr = row["atr"] if "atr" in row else np.nan
    if np.isnan(atr):
        atr = compute_atr(row.to_frame().T).iloc[-1]

    if prob >= buy_th:
        direction = "BUY"
        sl = price - atr_mult * atr
        tp = price + rr * (price - sl)

    elif prob <= sell_th:
        direction = "SELL"
        sl = price + atr_mult * atr
        tp = price - rr * (sl - price)

    else:
        return {"direction": "NEUTRAL", "entry": price, "stop": None, "tp": None}

    return {"direction": direction, "entry": price, "stop": sl, "tp": tp}

# train_model_M15_binary.py
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# ======================
# CONFIG (M15 OPTIMIZED)
# ======================
SEQ_LEN = 30
FUTURE_WINDOW = 8      # 🔥 shorter horizon for M15
ATR_MULT = 1.4         # 🔥 stronger breakout requirement

MODEL_OUT = "forex_model_M15_binary.keras"
SCALER_OUT = "forex_scaler_M15_binary.pkl"

# ======================
# LOAD CSV (MT5)
# ======================
def load_mt_csv(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['date','time','open','high','low','close','volume']

    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna().reset_index(drop=True)
    return df

# ======================
# INDICATORS
# ======================
def RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def ATR(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def ADX(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = low.diff().abs()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()

    return adx

# ======================
# FEATURES (MATCH LIVE BOT)
# ======================
def add_features(df):
    out = pd.DataFrame(index=df.index)

    out['close_ret'] = df['close'].pct_change().fillna(0)
    out['ema_dist'] = (df['close'] - df['close'].ewm(span=21).mean()) / df['close']
    out['hl_range'] = (df['high'] - df['low']) / df['close']
    out['volume'] = df['volume']
    out['ema_diff'] = df['close'].ewm(span=9).mean() - df['close'].ewm(span=21).mean()
    out['rsi14'] = RSI(df['close'])
    out['atr14'] = ATR(df)
    out['adx14'] = ADX(df)
    out['momentum'] = df['close'] - df['close'].shift(5)
    out['ema_trend'] = (
        df['close'].ewm(span=50).mean() >
        df['close'].ewm(span=200).mean()
    ).astype(int)


    out = out.bfill()
    return out, out.columns.tolist()

# ======================
# BINARY LABELS
# ======================
def make_labels(df):

    atr = ATR(df)
    labels = np.full(len(df), np.nan)

    for i in range(len(df) - FUTURE_WINDOW):

        if np.isnan(atr.iloc[i]):
            continue

        upper = df['close'].iloc[i] + atr.iloc[i] * ATR_MULT
        lower = df['close'].iloc[i] - atr.iloc[i] * ATR_MULT

        future_slice = df.iloc[i+1:i+1+FUTURE_WINDOW]

        hit_buy = (future_slice['high'] >= upper).any()
        hit_sell = (future_slice['low'] <= lower).any()

        if hit_buy and not hit_sell:
            labels[i] = 1  # BUY
        elif hit_sell and not hit_buy:
            labels[i] = 0  # SELL

        if i % 50000 == 0:
            print(f"Processing label {i}/{len(df)}")

    valid = ~np.isnan(labels)
    print("Binary distribution:", np.unique(labels[valid], return_counts=True))

    return labels

# ======================
# SEQUENCES
# ======================
def create_sequences(X, y):
    Xs, ys = [], []

    for i in range(len(X) - SEQ_LEN):
        if not np.isnan(y[i+SEQ_LEN-1]):
            Xs.append(X[i:i+SEQ_LEN])
            ys.append(y[i+SEQ_LEN-1])

    return np.array(Xs), np.array(ys)

def create_dual_sequences(X1, X2, y):
    Xs1, Xs2, ys = [], [], []

    for i in range(len(X1) - SEQ_LEN):
        if not np.isnan(y[i+SEQ_LEN-1]):
            Xs1.append(X1[i:i+SEQ_LEN])
            Xs2.append(X2[i:i+SEQ_LEN])
            ys.append(y[i+SEQ_LEN-1])

    return np.array(Xs1), np.array(Xs2), np.array(ys)


# ======================
# MODEL
# ======================
def build_dual_model(m1_steps, m1_features, m15_steps, m15_features):
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Conv1D, LSTM, Dense,
        Dropout, concatenate, BatchNormalization, MaxPooling1D
    )

    # ===== M1 branch =====
    input_m1 = Input(shape=(m1_steps, m1_features))

    x1 = Conv1D(64, 3, activation='relu')(input_m1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)

    x1 = LSTM(64, return_sequences=True)(x1)
    x1 = Dropout(0.3)(x1)

    x1 = LSTM(32)(x1)  # ⭐ CRITICAL — collapses to vector
    x1 = Dropout(0.3)(x1)

    # ===== M15 branch =====
    input_m15 = Input(shape=(m15_steps, m15_features))

    x2 = Conv1D(32, 3, activation='relu')(input_m15)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = LSTM(32)(x2)
    x2 = Dropout(0.3)(x2)
    

    # ===== Merge =====
    merged = concatenate([x2, x1])

    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_m1, input_m15], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1_csv", required=True)
    parser.add_argument("--m15_csv", required=True)

    args = parser.parse_args()

    print("📥 Loading data...")
    print("📥 Loading M1...")
    df_m1 = load_mt_csv(args.m1_csv)

    print("📥 Loading M15...")
    df_m15 = load_mt_csv(args.m15_csv)


    print("🧠 Adding M1 features...")
    X_m1_df, _ = add_features(df_m1)

    print("🧠 Adding M15 features...")
    X_m15_df, _ = add_features(df_m15)


    print("🏷️ Creating labels from M15...")
    labels = make_labels(df_m15)

    valid = ~np.isnan(labels)

    X_m15_df = X_m15_df[valid]
    labels = labels[valid]

    # align M1 length
    X_m1_df = X_m1_df.iloc[-len(X_m15_df):]


    valid = ~np.isnan(labels)
    labels = labels[valid]

    scaler_m1 = StandardScaler()
    scaler_m15 = StandardScaler()

    X_m1_scaled = scaler_m1.fit_transform(X_m1_df)
    X_m15_scaled = scaler_m15.fit_transform(X_m15_df)

    joblib.dump(scaler_m1, "scaler_m1.pkl")
    joblib.dump(scaler_m15, "scaler_m15.pkl")


    X_m1_seq, X_m15_seq, y_seq = create_dual_sequences(
        X_m1_scaled,
        X_m15_scaled,
        labels
    )


    split = int(len(X_m1_seq) * 0.8)

    X_m1_train, X_m1_test = X_m1_seq[:split], X_m1_seq[split:]
    X_m15_train, X_m15_test = X_m15_seq[:split], X_m15_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]


    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    model = build_dual_model(
        SEQ_LEN, X_m1_train.shape[2],
        SEQ_LEN, X_m15_train.shape[2]
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            mode='max',
            restore_best_weights=True,
            min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(MODEL_OUT, save_best_only=True)
    ]

    history = model.fit(
        [X_m1_train, X_m15_train],
        y_train,
        validation_data=([X_m1_test, X_m15_test], y_test),
        epochs=60,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks
    )


    model.save(MODEL_OUT)
    print("✅ M15 Binary Training complete")
    best_val = max(history.history['val_accuracy'])
    last_val = history.history['val_accuracy'][-1]

    print("Best:", best_val)
    print("Last:", last_val)

if __name__ == "__main__":
    main()

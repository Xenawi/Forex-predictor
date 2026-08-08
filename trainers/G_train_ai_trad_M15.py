import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, concatenate
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ======================
# CONFIG (M15 OPTIMIZED)
# FIX #3: Separate sequence lengths to match real time coverage
#         M1: 120 bars = ~2 hours
#         M15: 40 bars  = ~10 hours (same real-time window)
# ======================
SEQ_LEN_M1  = 120
SEQ_LEN_M15 = 40

# FIX #2: Lower ATR multiplier + longer future window
#         Gives price enough room and time to move, generating
#         more valid labels instead of starving the model.
FUTURE_WINDOW = 12    # was 8  → now 3 hours on M15
ATR_MULT      = 0.9   # was 1.4 → lower bar for a valid signal

MODEL_OUT      = "forex_model_M15_binary.keras"
SCALER_M1_OUT  = "scaler_m1.pkl"
SCALER_M15_OUT = "scaler_m15.pkl"


# ======================
# LOAD CSV (MT5)
# ======================
def load_mt_csv(path):
    print(f"📖 Reading {path}...")
    df = pd.read_csv(path, header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    # Remove duplicate timestamps
    initial_len = len(df)
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    if initial_len != len(df):
        print(f"⚠️  Removed {initial_len - len(df)} duplicate rows from {path}")

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna().set_index('datetime').sort_index()
    return df


# ======================
# INDICATORS
# ======================
def RSI(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs       = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def ATR(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# FIX #4: Corrected ADX — proper +DM/-DM logic
def ADX(df, period=14):
    high  = df['high']
    low   = df['low']
    close = df['close']

    # Raw directional moves
    up_move   = high.diff()
    down_move = (-low.diff())

    # +DM: up move only when it exceeds down move AND is positive
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    # -DM: down move only when it exceeds up move AND is positive
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_val  = tr.rolling(period).mean()
    plus_di  = 100 * (plus_dm.rolling(period).mean()  / (atr_val + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr_val + 1e-9))

    dx  = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    adx = dx.rolling(period).mean()
    return adx


# ======================
# FEATURES
# ======================
def add_features(df):
    out = pd.DataFrame(index=df.index)
    out['close_ret']  = df['close'].pct_change().fillna(0)
    out['ema_dist']   = (df['close'] - df['close'].ewm(span=21).mean()) / df['close']
    out['hl_range']   = (df['high'] - df['low']) / df['close']
    out['volume']     = df['volume']
    out['ema_diff']   = df['close'].ewm(span=9).mean() - df['close'].ewm(span=21).mean()
    out['rsi14']      = RSI(df['close'])
    out['atr14']      = ATR(df)
    out['adx14']      = ADX(df)        # now using fixed ADX
    out['momentum']   = df['close'] - df['close'].shift(5)
    out['ema_trend']  = (
        df['close'].ewm(span=50).mean() > df['close'].ewm(span=200).mean()
    ).astype(int)

    # Forward fill to prevent any data leakage
    out = out.ffill().fillna(0)
    return out


# ======================
# BINARY LABELS
# FIX #1: Track and log discarded ambiguous candles
# ======================
def make_labels(df):
    atr    = ATR(df)
    labels = pd.Series(index=df.index, dtype=float)

    discarded = 0
    buy_count  = 0
    sell_count = 0

    for i in range(len(df) - FUTURE_WINDOW):
        if pd.isna(atr.iloc[i]):
            continue

        upper = df['close'].iloc[i] + atr.iloc[i] * ATR_MULT
        lower = df['close'].iloc[i] - atr.iloc[i] * ATR_MULT

        future_slice = df.iloc[i + 1: i + 1 + FUTURE_WINDOW]
        hit_buy  = (future_slice['high'] >= upper).any()
        hit_sell = (future_slice['low']  <= lower).any()

        if hit_buy and not hit_sell:
            labels.iloc[i] = 1.0   # BUY
            buy_count += 1
        elif hit_sell and not hit_buy:
            labels.iloc[i] = 0.0   # SELL
            sell_count += 1
        elif hit_buy and hit_sell:
            discarded += 1          # Ambiguous — skip cleanly

        if i % 50_000 == 0:
            print(f"  Label progress: {i}/{len(df)}")

    total_labeled = buy_count + sell_count
    print(f"\n📊 Label Summary:")
    print(f"   BUY  : {buy_count:,}  ({100*buy_count/max(total_labeled,1):.1f}%)")
    print(f"   SELL : {sell_count:,}  ({100*sell_count/max(total_labeled,1):.1f}%)")
    print(f"   Ambiguous (discarded): {discarded:,}")
    print(f"   Total valid labels   : {total_labeled:,}\n")
    return labels


# ======================
# SYNCED SEQUENCES
# FIX #3: Use separate SEQ_LEN per timeframe
# ======================
def create_aligned_sequences(X_m1, X_m15, labels, valid_times):
    Xs1, Xs2, ys = [], [], []

    m1_indices  = X_m1.index.get_indexer(valid_times,  method='ffill')
    m15_indices = X_m15.index.get_indexer(valid_times, method='ffill')

    for t, idx_m1, idx_m15 in zip(valid_times, m1_indices, m15_indices):
        if idx_m1 < SEQ_LEN_M1 - 1 or idx_m15 < SEQ_LEN_M15 - 1:
            continue

        seq_m1  = X_m1.iloc[idx_m1  - SEQ_LEN_M1  + 1: idx_m1  + 1].values
        seq_m15 = X_m15.iloc[idx_m15 - SEQ_LEN_M15 + 1: idx_m15 + 1].values

        Xs1.append(seq_m1)
        Xs2.append(seq_m15)
        ys.append(labels.loc[t])

    return np.array(Xs1), np.array(Xs2), np.array(ys)


# ======================
# MODEL
# FIX #5: Balanced branch sizes — M15 context gets equal weight
# ======================
def build_dual_model(m1_steps, m1_features, m15_steps, m15_features):

    # ── M1 branch (short-term momentum / entry timing) ──
    input_m1 = Input(shape=(m1_steps, m1_features), name="input_m1")
    x1 = Conv1D(64, 3, activation='relu', padding='same')(input_m1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = LSTM(64, return_sequences=True)(x1)
    x1 = Dropout(0.3)(x1)
    x1 = LSTM(32)(x1)
    x1 = Dropout(0.3)(x1)

    # ── M15 branch (macro direction / trend context) ──
    # FIX #5: Increased to 64 units so M15 isn't underrepresented in merge
    input_m15 = Input(shape=(m15_steps, m15_features), name="input_m15")
    x2 = Conv1D(64, 3, activation='relu', padding='same')(input_m15)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = LSTM(64, return_sequences=True)(x2)
    x2 = Dropout(0.3)(x2)
    x2 = LSTM(32)(x2)
    x2 = Dropout(0.3)(x2)

    # ── Merge (M15 first so macro context leads) ──
    merged = concatenate([x2, x1])
    x = Dense(64, activation='relu')(merged)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_m1, input_m15], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0008),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ======================
# MAIN
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1_csv",  required=True, help="Path to M1 CSV")
    parser.add_argument("--m15_csv", required=True, help="Path to M15 CSV")
    args = parser.parse_args()

    print("📥 Loading M1 Data...")
    df_m1  = load_mt_csv(args.m1_csv)
    print("📥 Loading M15 Data...")
    df_m15 = load_mt_csv(args.m15_csv)

    print("🧠 Generating Features...")
    X_m1_df  = add_features(df_m1)
    X_m15_df = add_features(df_m15)

    print("🏷️  Generating Target Labels (from M15)...")
    labels = make_labels(df_m15)

    valid_times = labels.dropna().index.sort_values()

    print("✂️  Splitting Train/Test (chronologically)...")
    split_idx  = int(len(valid_times) * 0.8)
    train_times = valid_times[:split_idx]
    test_times  = valid_times[split_idx:]

    # FIX #6: Strict boundary — exclude split timestamp itself from scaler fit
    split_time = valid_times[split_idx] - pd.Timedelta(minutes=1)

    print("⚖️  Fitting Scalers (strictly on train data only)...")
    scaler_m1  = StandardScaler()
    scaler_m15 = StandardScaler()

    scaler_m1.fit(X_m1_df.loc[:split_time])
    scaler_m15.fit(X_m15_df.loc[:split_time])

    joblib.dump(scaler_m1,  SCALER_M1_OUT)
    joblib.dump(scaler_m15, SCALER_M15_OUT)
    print(f"💾 Scalers saved → {SCALER_M1_OUT}, {SCALER_M15_OUT}")

    X_m1_scaled  = pd.DataFrame(
        scaler_m1.transform(X_m1_df),
        index=X_m1_df.index, columns=X_m1_df.columns
    )
    X_m15_scaled = pd.DataFrame(
        scaler_m15.transform(X_m15_df),
        index=X_m15_df.index, columns=X_m15_df.columns
    )

    print("🔄 Building Synced Sequences...")
    X_m1_train, X_m15_train, y_train = create_aligned_sequences(
        X_m1_scaled, X_m15_scaled, labels, train_times
    )
    X_m1_test, X_m15_test, y_test = create_aligned_sequences(
        X_m1_scaled, X_m15_scaled, labels, test_times
    )

    print(f"📊 Training Samples : {len(y_train):,}")
    print(f"📊 Test Samples     : {len(y_test):,}")

    # Class weights for imbalanced labels
    class_weights      = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"⚖️  Class weights: {class_weights_dict}")

    print("🏗️  Building Model...")
    model = build_dual_model(
        SEQ_LEN_M1,  X_m1_train.shape[2],
        SEQ_LEN_M15, X_m15_train.shape[2]
    )
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', patience=8, mode='max',
            restore_best_weights=True, min_delta=0.001
        ),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-5, verbose=1
        ),
        ModelCheckpoint(MODEL_OUT, save_best_only=True, monitor='val_accuracy')
    ]

    print("🚀 Training Model...")
    history = model.fit(
        [X_m1_train, X_m15_train], y_train,
        validation_data=([X_m1_test, X_m15_test], y_test),
        epochs=60,
        batch_size=64,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )

    model.save(MODEL_OUT)
    print(f"\n✅ Training Complete! Model saved → {MODEL_OUT}")
    print(f"   Best Val Accuracy : {max(history.history['val_accuracy']):.4f}")
    print(f"   Last Val Accuracy : {history.history['val_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    main()
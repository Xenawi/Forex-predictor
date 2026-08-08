"""
train_forex_v2.py
=================
Upgrades over v1:
  1. Transformer encoder branches (M1 + M15 + H1)
  2. 3-class labels  (0=SELL | 1=NEUTRAL | 2=BUY)
  3. Walk-forward validation with purge + embargo
  4. Ensemble training (N models, different seeds)
  5. Kelly position-sizing output head
  6. Corrected ADX
  7. Regime detection sub-model (trending / ranging)
  8. Session + order-flow proxy features

Usage:
    python train_forex_v2.py \
        --m1_csv  EURUSD_M1.csv  \
        --m15_csv EURUSD_M15.csv \
        --h1_csv  EURUSD_H1.csv
"""

import argparse
import random
import os

import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, concatenate, Add
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEQ_LEN_M1  = 120   # ~2 hours
SEQ_LEN_M15 = 40    # ~10 hours
SEQ_LEN_H1  = 24    # ~1 day

FUTURE_WINDOW = 12   # bars on M15 (~3 hours)
ATR_MULT      = 0.9

# Walk-forward
N_FOLDS       = 5        # number of walk-forward windows
EMBARGO_BARS  = 20       # M15 bars to purge around each boundary
PURGE_BARS    = 10

# Ensemble
N_MODELS      = 3

# Transformer
N_HEADS       = 4
FF_DIM        = 64       # feed-forward dim inside transformer block
TRANSFORMER_LAYERS = 2

# Output paths
MODEL_DIR      = "ensemble_models"
SCALER_M1_OUT  = "scaler_m1.pkl"
SCALER_M15_OUT = "scaler_m15.pkl"
SCALER_H1_OUT  = "scaler_h1.pkl"
REGIME_OUT     = "regime_model.keras"
KELLY_OUT      = "kelly_model.keras"

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# REPRODUCIBILITY SEED HELPER
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────
# LOAD CSV  (MT5 format)
# ─────────────────────────────────────────────
def load_mt_csv(path: str) -> pd.DataFrame:
    print(f"📖  Reading {path} ...")
    df = pd.read_csv(path, header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    before = len(df)
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    removed = before - len(df)
    if removed:
        print(f"   ⚠️  Removed {removed:,} duplicate rows")

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.dropna().set_index('datetime').sort_index()


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0).rolling(period).mean()
    loss     = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ADX(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Corrected ADX — proper +DM / -DM masking."""
    up   = df['high'].diff()
    down = (-df['low'].diff())
    plus_dm  = up.where((up > down)   & (up > 0),   0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low']  - df['close'].shift()).abs()
    ], axis=1).max(axis=1).rolling(period).mean()

    plus_di  = 100 * plus_dm.rolling(period).mean()  / (tr + 1e-9)
    minus_di = 100 * minus_dm.rolling(period).mean() / (tr + 1e-9)
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100
    return dx.rolling(period).mean()


# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Price features
    out['close_ret']     = df['close'].pct_change().fillna(0)
    out['ema_dist']      = (df['close'] - df['close'].ewm(span=21).mean()) / df['close']
    out['hl_range']      = (df['high'] - df['low']) / df['close']
    out['ema_diff']      = df['close'].ewm(span=9).mean() - df['close'].ewm(span=21).mean()
    out['momentum']      = df['close'] - df['close'].shift(5)
    out['ema_trend']     = (
        df['close'].ewm(span=50).mean() > df['close'].ewm(span=200).mean()
    ).astype(int)

    # Indicators
    out['rsi14']         = RSI(df['close'])
    out['atr14']         = ATR(df)
    out['adx14']         = ADX(df)

    # ── ORDER FLOW PROXIES ──
    # Where did close land inside the bar? (1=full buy pressure, 0=full sell)
    out['buy_pressure']  = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    out['vol_delta']     = df['volume'].diff().fillna(0)
    out['volume']        = df['volume']

    # ── SESSION FEATURE ──
    # 0=Asian  1=London  2=NY  3=Off
    hour = df.index.hour
    session = np.where(hour < 7, 0,
              np.where(hour < 13, 1,
              np.where(hour < 20, 2, 3)))
    out['session']       = session

    out = out.ffill().fillna(0)
    return out


# ─────────────────────────────────────────────
# 3-CLASS LABELS
#   0 = SELL  |  1 = NEUTRAL  |  2 = BUY
# ─────────────────────────────────────────────
def make_labels(df: pd.DataFrame) -> pd.Series:
    atr    = ATR(df)
    labels = pd.Series(index=df.index, dtype=float)

    counts = {0: 0, 1: 0, 2: 0, 'ambig': 0}

    for i in range(len(df) - FUTURE_WINDOW):
        if pd.isna(atr.iloc[i]):
            continue

        upper        = df['close'].iloc[i] + atr.iloc[i] * ATR_MULT
        lower        = df['close'].iloc[i] - atr.iloc[i] * ATR_MULT
        future_slice = df.iloc[i + 1: i + 1 + FUTURE_WINDOW]

        hit_buy  = (future_slice['high'] >= upper).any()
        hit_sell = (future_slice['low']  <= lower).any()

        if hit_buy and not hit_sell:
            labels.iloc[i] = 2.0    # BUY
            counts[2] += 1
        elif hit_sell and not hit_buy:
            labels.iloc[i] = 0.0    # SELL
            counts[0] += 1
        elif not hit_buy and not hit_sell:
            labels.iloc[i] = 1.0    # NEUTRAL
            counts[1] += 1
        else:
            counts['ambig'] += 1    # Both hit — discard

        if i % 50_000 == 0:
            print(f"   Labels progress {i:,}/{len(df):,}")

    total = counts[0] + counts[1] + counts[2]
    print(f"\n📊  Label summary (total valid: {total:,})")
    print(f"   SELL    : {counts[0]:>8,}  ({100*counts[0]/max(total,1):.1f}%)")
    print(f"   NEUTRAL : {counts[1]:>8,}  ({100*counts[1]/max(total,1):.1f}%)")
    print(f"   BUY     : {counts[2]:>8,}  ({100*counts[2]/max(total,1):.1f}%)")
    print(f"   Ambig   : {counts['ambig']:>8,}  (discarded)")
    return labels


# ─────────────────────────────────────────────
# REGIME LABELS  (for auxiliary regime model)
#   0 = RANGING   |   1 = TRENDING
# ─────────────────────────────────────────────
def make_regime_labels(df: pd.DataFrame, period: int = 14) -> pd.Series:
    adx = ADX(df, period)
    return (adx > 25).astype(float)


# ─────────────────────────────────────────────
# WALK-FORWARD FOLDS  with purge + embargo
# ─────────────────────────────────────────────
def make_walk_forward_folds(valid_times: pd.DatetimeIndex, n_folds: int):
    """
    Returns list of (train_times, test_times) tuples.
    Samples within EMBARGO_BARS of any boundary are removed.
    """
    total      = len(valid_times)
    fold_size  = total // (n_folds + 1)
    folds      = []

    for k in range(1, n_folds + 1):
        train_end_idx  = k * fold_size
        test_start_idx = train_end_idx + PURGE_BARS
        test_end_idx   = min(test_start_idx + fold_size, total)

        if test_start_idx >= total:
            break

        # Embargo: drop EMBARGO_BARS on both sides of boundary
        train_times = valid_times[: max(0, train_end_idx - EMBARGO_BARS)]
        test_times  = valid_times[min(total - 1, test_start_idx + EMBARGO_BARS): test_end_idx]

        folds.append((train_times, test_times))
        print(f"   Fold {k}: train={len(train_times):,}  test={len(test_times):,}")

    return folds


# ─────────────────────────────────────────────
# SEQUENCE BUILDER  (3 timeframes)
# ─────────────────────────────────────────────
def create_sequences(X_m1, X_m15, X_h1, labels, valid_times):
    Xs1, Xs2, Xs3, ys = [], [], [], []

    idx_m1  = X_m1.index.get_indexer(valid_times,  method='ffill')
    idx_m15 = X_m15.index.get_indexer(valid_times, method='ffill')
    idx_h1  = X_h1.index.get_indexer(valid_times,  method='ffill')

    for t, i1, i15, ih in zip(valid_times, idx_m1, idx_m15, idx_h1):
        if i1 < SEQ_LEN_M1 - 1 or i15 < SEQ_LEN_M15 - 1 or ih < SEQ_LEN_H1 - 1:
            continue
        Xs1.append(X_m1.iloc[i1  - SEQ_LEN_M1  + 1: i1  + 1].values)
        Xs2.append(X_m15.iloc[i15 - SEQ_LEN_M15 + 1: i15 + 1].values)
        Xs3.append(X_h1.iloc[ih  - SEQ_LEN_H1  + 1: ih  + 1].values)
        ys.append(labels.loc[t])

    return (np.array(Xs1, dtype=np.float32),
            np.array(Xs2, dtype=np.float32),
            np.array(Xs3, dtype=np.float32),
            np.array(ys,  dtype=np.float32))


# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────
def transformer_block(x, embed_dim: int, n_heads: int, ff_dim: int, dropout: float = 0.1):
    """One Transformer encoder layer with residual connections."""
    # Multi-head self-attention
    attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=embed_dim // n_heads)(x, x)
    attn_out = Dropout(dropout)(attn_out)
    x        = LayerNormalization(epsilon=1e-6)(Add()([x, attn_out]))

    # Feed-forward
    ff = Dense(ff_dim,    activation='relu')(x)
    ff = Dense(embed_dim, activation='linear')(ff)
    ff = Dropout(dropout)(ff)
    x  = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x


def build_branch(seq_len: int, n_features: int,
                 embed_dim: int = 64,
                 name: str = "branch") -> tuple:
    """
    Single Transformer encoder branch.
    Returns (input_tensor, output_tensor).
    """
    inp = Input(shape=(seq_len, n_features), name=f"input_{name}")

    # Project to embed_dim
    x = Dense(embed_dim, activation='linear', name=f"proj_{name}")(inp)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Stack transformer layers
    for _ in range(TRANSFORMER_LAYERS):
        x = transformer_block(x, embed_dim, N_HEADS, FF_DIM, dropout=0.1)

    # Pool across time
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    return inp, x



# ─────────────────────────────────────────────
# DIRECTION MODEL  (single output)
# Single-output so Keras class_weight works fine
# ─────────────────────────────────────────────
def build_model(m1_feat: int, m15_feat: int, h1_feat: int) -> Model:
    inp_m1,  x1 = build_branch(SEQ_LEN_M1,  m1_feat,  embed_dim=64, name="m1")
    inp_m15, x2 = build_branch(SEQ_LEN_M15, m15_feat, embed_dim=64, name="m15")
    inp_h1,  x3 = build_branch(SEQ_LEN_H1,  h1_feat,  embed_dim=32, name="h1")

    merged = concatenate([x3, x2, x1], name="merge")
    trunk  = Dense(128, activation='relu')(merged)
    trunk  = Dropout(0.3)(trunk)
    trunk  = Dense(64,  activation='relu')(trunk)
    trunk  = Dropout(0.3)(trunk)
    out    = Dense(3, activation='softmax', name='direction')(trunk)

    model = Model(inputs=[inp_m1, inp_m15, inp_h1], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# KELLY MODEL  (separate single-output model)
# Trained on M15 sequences to predict position
# size fraction [0,1] based on rolling edge.
# ─────────────────────────────────────────────
def build_kelly_model(m15_feat: int) -> Model:
    inp, x = build_branch(SEQ_LEN_M15, m15_feat, embed_dim=32, name="kelly")
    x   = Dense(32, activation='relu')(x)
    out = Dense(1,  activation='sigmoid', name='kelly')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model


# ─────────────────────────────────────────────
# REGIME MODEL  (simple, separate)
# ─────────────────────────────────────────────
def build_regime_model(seq_len: int, n_features: int) -> Model:
    inp, x = build_branch(seq_len, n_features, embed_dim=32, name="regime")
    out     = Dense(1, activation='sigmoid', name='regime_out')(x)
    model   = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# KELLY TARGET COMPUTATION
# ─────────────────────────────────────────────
def make_kelly_targets(y_labels: np.ndarray, window: int = 200) -> np.ndarray:
    kelly = np.zeros(len(y_labels), dtype=np.float32)
    for i in range(len(y_labels)):
        start       = max(0, i - window)
        segment     = y_labels[start:i + 1]
        non_neutral = segment[segment != 1]
        if len(non_neutral) == 0:
            kelly[i] = 0.0
            continue
        win_rate = np.mean(non_neutral == 2)
        edge     = abs(win_rate - 0.5)
        fraction = edge / (1 - edge + 1e-9)
        kelly[i] = float(np.clip(fraction, 0.0, 1.0))
    return kelly


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1_csv",  required=True)
    parser.add_argument("--m15_csv", required=True)
    parser.add_argument("--h1_csv",  required=True)
    parser.add_argument("--n_models", type=int, default=N_MODELS,
                        help="Ensemble size (default 3)")
    args = parser.parse_args()

    print("\n=== Loading Data ===")
    df_m1  = load_mt_csv(args.m1_csv)
    df_m15 = load_mt_csv(args.m15_csv)
    df_h1  = load_mt_csv(args.h1_csv)

    print("\n=== Generating Features ===")
    X_m1_df  = add_features(df_m1)
    X_m15_df = add_features(df_m15)
    X_h1_df  = add_features(df_h1)
    n_feat   = X_m1_df.shape[1]
    print(f"   Features per timeframe: {n_feat}")

    print("\n=== Generating Labels ===")
    labels      = make_labels(df_m15)
    valid_times = labels.dropna().index.sort_values()

    regime_labels = make_regime_labels(df_m15)

    print("\n=== Walk-Forward Folds ===")
    folds = make_walk_forward_folds(valid_times, N_FOLDS)

    global_split = valid_times[int(len(valid_times) * 0.6)]
    split_strict = global_split - pd.Timedelta(minutes=1)
    print(f"\n=== Fitting Scalers (up to {split_strict.date()}) ===")

    scaler_m1  = StandardScaler().fit(X_m1_df.loc[:split_strict])
    scaler_m15 = StandardScaler().fit(X_m15_df.loc[:split_strict])
    scaler_h1  = StandardScaler().fit(X_h1_df.loc[:split_strict])

    joblib.dump(scaler_m1,  SCALER_M1_OUT)
    joblib.dump(scaler_m15, SCALER_M15_OUT)
    joblib.dump(scaler_h1,  SCALER_H1_OUT)
    print("   Scalers saved.")

    X_m1_sc  = pd.DataFrame(scaler_m1.transform(X_m1_df),
                             index=X_m1_df.index, columns=X_m1_df.columns)
    X_m15_sc = pd.DataFrame(scaler_m15.transform(X_m15_df),
                             index=X_m15_df.index, columns=X_m15_df.columns)
    X_h1_sc  = pd.DataFrame(scaler_h1.transform(X_h1_df),
                             index=X_h1_df.index, columns=X_h1_df.columns)

    # ── Regime model ──────────────────────────
    print("\n=== Training Regime Model ===")
    _, X_m15_reg, _, y_reg = create_sequences(
        X_m1_sc, X_m15_sc, X_h1_sc, regime_labels, valid_times
    )
    split_reg    = int(len(y_reg) * 0.8)
    regime_model = build_regime_model(SEQ_LEN_M15, n_feat)
    regime_model.fit(
        X_m15_reg[:split_reg], y_reg[:split_reg],
        validation_data=(X_m15_reg[split_reg:], y_reg[split_reg:]),
        epochs=20, batch_size=128,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1
    )
    regime_model.save(REGIME_OUT)
    print(f"   Regime model saved -> {REGIME_OUT}")

    # ── Kelly model ───────────────────────────
    print("\n=== Training Kelly Model ===")
    _, X_m15_all, _, y_all = create_sequences(
        X_m1_sc, X_m15_sc, X_h1_sc, labels, valid_times
    )
    kelly_all   = make_kelly_targets(y_all)
    split_k     = int(len(y_all) * 0.8)
    kelly_model = build_kelly_model(n_feat)
    kelly_model.fit(
        X_m15_all[:split_k], kelly_all[:split_k],
        validation_data=(X_m15_all[split_k:], kelly_all[split_k:]),
        epochs=20, batch_size=128,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1
    )
    kelly_model.save(KELLY_OUT)
    print(f"   Kelly model saved -> {KELLY_OUT}")

    # ─────────────────────────────────────────
    # ENSEMBLE x WALK-FORWARD
    # Single-output -> class_weight works fine
    # ─────────────────────────────────────────
    print(f"\n=== Ensemble Training ({args.n_models} models x {len(folds)} folds) ===")

    fold_results = []

    for fold_idx, (train_times, test_times) in enumerate(folds):
        print(f"\n-- Fold {fold_idx + 1}/{len(folds)} --")

        X1_tr, X2_tr, X3_tr, y_tr = create_sequences(
            X_m1_sc, X_m15_sc, X_h1_sc, labels, train_times
        )
        X1_te, X2_te, X3_te, y_te = create_sequences(
            X_m1_sc, X_m15_sc, X_h1_sc, labels, test_times
        )
        print(f"   train={len(y_tr):,}  test={len(y_te):,}")

        if len(y_tr) < 200 or len(y_te) < 50:
            print("   WARNING: Too few samples, skipping fold")
            continue

        cw = compute_class_weight('balanced',
                                  classes=np.unique(y_tr), y=y_tr)
        cw_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_tr), cw)}
        print(f"   Class weights: {cw_dict}")

        fold_model_paths = []

        for m_idx in range(args.n_models):
            seed = 42 + fold_idx * 100 + m_idx
            set_seed(seed)

            model_path = os.path.join(
                MODEL_DIR, f"model_fold{fold_idx+1}_m{m_idx+1}.keras"
            )
            print(f"\n   Model {m_idx+1}/{args.n_models}  (seed={seed})")
            model = build_model(n_feat, n_feat, n_feat)

            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy', patience=7, mode='max',
                    restore_best_weights=True, min_delta=0.001
                ),
                ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5,
                    patience=4, min_lr=1e-5, verbose=0
                ),
                ModelCheckpoint(
                    model_path, save_best_only=True, monitor='val_accuracy'
                )
            ]

            history = model.fit(
                [X1_tr, X2_tr, X3_tr], y_tr,
                validation_data=([X1_te, X2_te, X3_te], y_te),
                epochs=60,
                batch_size=64,
                class_weight=cw_dict,
                callbacks=callbacks,
                verbose=1
            )

            best_acc = max(history.history['val_accuracy'])
            print(f"   Best val_accuracy: {best_acc:.4f}")
            fold_model_paths.append((model_path, best_acc))

        fold_results.append({'fold': fold_idx + 1, 'models': fold_model_paths})

    # ── Summary ───────────────────────────────
    print("\n=======================================")
    print("  TRAINING COMPLETE - ENSEMBLE SUMMARY")
    print("=======================================")
    all_accs = []
    for fr in fold_results:
        for path, acc in fr['models']:
            all_accs.append(acc)
            print(f"  Fold {fr['fold']}  {os.path.basename(path):35s}  acc={acc:.4f}")
    if all_accs:
        print(f"\n  Mean val accuracy : {np.mean(all_accs):.4f}")
        print(f"  Best val accuracy : {max(all_accs):.4f}")

    print(f"\n  Ensemble models -> {MODEL_DIR}/")
    print(f"  Regime model    -> {REGIME_OUT}")
    print(f"  Kelly model     -> {KELLY_OUT}")
    print(f"  Scalers         -> scaler_m1/m15/h1.pkl")


if __name__ == "__main__":
    main()

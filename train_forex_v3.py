"""
train_forex_v3.py
=================
Clean two-timeframe model: H1 (trend context) + M15 (entry signal)
Trades/acts on M15 bars.

Improvements over v2:
  - Removed M1 branch (was noise for 3hr horizon, main speed bottleneck)
  - Simpler Transformer (1 layer, smaller embed) — faster, less overfitting
  - 3-class labels: 0=SELL | 1=NEUTRAL | 2=BUY
  - Walk-forward validation with purge + embargo
  - Ensemble training (N models, different seeds)
  - Separate Kelly sizing model
  - Separate Regime detection model
  - Recency-weighted sample weights (recent bars matter more)

Usage:
    python train_forex_v3.py \
        --m15_csv EURUSD_M15.csv \
        --h1_csv  EURUSD_H1.csv

Colab one-liner:
    !python train_forex_v3.py --m15_csv "/content/drive/MyDrive/EURUSD_M15.csv" \
                               --h1_csv  "/content/drive/MyDrive/EURUSD_H1.csv"
"""

import argparse
import random
import os
import joblib
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
    concatenate, Add
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
SEQ_LEN_M15 = 48     # 12 hours of M15 context
SEQ_LEN_H1  = 24     # 24 hours of H1 context (1 full day)

FUTURE_WINDOW = 12   # 12 M15 bars = 3 hours ahead
ATR_MULT      = 0.9  # ATR multiplier for label generation

# Transformer
N_HEADS            = 4
FF_DIM             = 64
TRANSFORMER_LAYERS = 1   # single layer — fast + avoids overfitting on small data

# Walk-forward
N_FOLDS      = 5
EMBARGO_BARS = 20
PURGE_BARS   = 10

# Ensemble
N_MODELS = 3

# Output paths
MODEL_DIR      = "ensemble_models"
SCALER_M15_OUT = "scaler_m15.pkl"
SCALER_H1_OUT  = "scaler_h1.pkl"
REGIME_OUT     = "regime_model.keras"
KELLY_OUT      = "kelly_model.keras"

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ─────────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────────
def load_mt_csv(path: str) -> pd.DataFrame:
    print(f"  Reading {path} ...")
    df = pd.read_csv(path, header=None)
    df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    before = len(df)
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    removed = before - len(df)
    if removed:
        print(f"  Removed {removed:,} duplicate rows")

    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna().set_index('datetime').sort_index()

    # Filter out low-liquidity holiday periods
    df = df[~((df.index.month == 12) & (df.index.day >= 24))]
    df = df[~((df.index.month == 1)  & (df.index.day <= 3))]

    print(f"  Loaded {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────
def RSI(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - (100 / (1 + gain / (loss + 1e-9)))


def ATR(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def ADX(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up       = df['high'].diff()
    down     = (-df['low'].diff())
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

    # Price structure
    out['close_ret']    = df['close'].pct_change().fillna(0)
    out['ema_dist']     = (df['close'] - df['close'].ewm(span=21).mean()) / df['close']
    out['hl_range']     = (df['high'] - df['low']) / df['close']
    out['ema_diff']     = (df['close'].ewm(span=9).mean()
                          - df['close'].ewm(span=21).mean())
    out['momentum']     = df['close'] - df['close'].shift(5)
    out['ema_trend']    = (
        df['close'].ewm(span=50).mean() > df['close'].ewm(span=200).mean()
    ).astype(int)

    # Indicators
    out['rsi14']        = RSI(df['close'])
    out['atr14']        = ATR(df)
    out['adx14']        = ADX(df)

    # Order flow proxies
    out['buy_pressure'] = (
        (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    )
    out['vol_delta']    = df['volume'].diff().fillna(0)
    out['volume']       = df['volume']

    # Session (0=Asian 1=London 2=NY 3=Off)
    hour = df.index.hour
    out['session'] = np.where(hour < 7,  0,
                    np.where(hour < 13, 1,
                    np.where(hour < 20, 2, 3)))

    out = out.ffill().fillna(0)
    return out


# ─────────────────────────────────────────────
# 3-CLASS LABELS
# 0=SELL | 1=NEUTRAL | 2=BUY
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
        hit_buy      = (future_slice['high'] >= upper).any()
        hit_sell     = (future_slice['low']  <= lower).any()

        if hit_buy and not hit_sell:
            labels.iloc[i] = 2.0
            counts[2] += 1
        elif hit_sell and not hit_buy:
            labels.iloc[i] = 0.0
            counts[0] += 1
        elif not hit_buy and not hit_sell:
            labels.iloc[i] = 1.0
            counts[1] += 1
        else:
            counts['ambig'] += 1

        if i % 20_000 == 0:
            print(f"  Labels: {i:,}/{len(df):,}")

    total = counts[0] + counts[1] + counts[2]
    print(f"\n  Label summary (total: {total:,})")
    print(f"  SELL    : {counts[0]:>7,}  ({100*counts[0]/max(total,1):.1f}%)")
    print(f"  NEUTRAL : {counts[1]:>7,}  ({100*counts[1]/max(total,1):.1f}%)")
    print(f"  BUY     : {counts[2]:>7,}  ({100*counts[2]/max(total,1):.1f}%)")
    print(f"  Ambig   : {counts['ambig']:>7,}  (discarded)")
    return labels


# ─────────────────────────────────────────────
# REGIME LABELS  (for regime model)
# 0=RANGING | 1=TRENDING
# ─────────────────────────────────────────────
def make_regime_labels(df: pd.DataFrame) -> pd.Series:
    return (ADX(df) > 25).astype(float)


# ─────────────────────────────────────────────
# RECENCY WEIGHTS
# Recent bars get higher weight during training
# ─────────────────────────────────────────────
def make_recency_weights(valid_times: pd.DatetimeIndex,
                         min_weight: float = 0.4) -> np.ndarray:
    n       = len(valid_times)
    weights = np.linspace(min_weight, 1.0, n).astype(np.float32)

    # Down-weight 2022 (extreme outlier year — rate hike spike)
    for i, t in enumerate(valid_times):
        if t.year == 2022:
            weights[i] *= 0.6

    return weights


# ─────────────────────────────────────────────
# WALK-FORWARD FOLDS
# ─────────────────────────────────────────────
def make_walk_forward_folds(valid_times: pd.DatetimeIndex, n_folds: int):
    total     = len(valid_times)
    fold_size = total // (n_folds + 1)
    folds     = []

    for k in range(1, n_folds + 1):
        train_end  = k * fold_size
        test_start = train_end + PURGE_BARS
        test_end   = min(test_start + fold_size, total)

        if test_start >= total:
            break

        train_times = valid_times[: max(0, train_end - EMBARGO_BARS)]
        test_times  = valid_times[
            min(total - 1, test_start + EMBARGO_BARS): test_end
        ]
        folds.append((train_times, test_times))
        print(f"  Fold {k}: train={len(train_times):,}  test={len(test_times):,}")

    return folds


# ─────────────────────────────────────────────
# SEQUENCE BUILDER  (M15 + H1 only)
# ─────────────────────────────────────────────
def create_sequences(X_m15, X_h1, labels, valid_times):
    Xs_m15, Xs_h1, ys = [], [], []

    idx_m15 = X_m15.index.get_indexer(valid_times, method='ffill')
    idx_h1  = X_h1.index.get_indexer(valid_times,  method='ffill')

    for t, i15, ih in zip(valid_times, idx_m15, idx_h1):
        if i15 < SEQ_LEN_M15 - 1 or ih < SEQ_LEN_H1 - 1:
            continue
        Xs_m15.append(X_m15.iloc[i15 - SEQ_LEN_M15 + 1: i15 + 1].values)
        Xs_h1.append(X_h1.iloc[ih  - SEQ_LEN_H1  + 1: ih  + 1].values)
        ys.append(labels.loc[t])

    return (np.array(Xs_m15, dtype=np.float32),
            np.array(Xs_h1,  dtype=np.float32),
            np.array(ys,     dtype=np.float32))


# ─────────────────────────────────────────────
# TRANSFORMER BLOCK
# ─────────────────────────────────────────────
def transformer_block(x, embed_dim: int, n_heads: int,
                      ff_dim: int, dropout: float = 0.1):
    attn = MultiHeadAttention(
        num_heads=n_heads, key_dim=embed_dim // n_heads
    )(x, x)
    attn = Dropout(dropout)(attn)
    x    = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    ff   = Dense(ff_dim,    activation='relu')(x)
    ff   = Dense(embed_dim, activation='linear')(ff)
    ff   = Dropout(dropout)(ff)
    x    = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    return x


def build_branch(seq_len: int, n_features: int,
                 embed_dim: int = 32, name: str = "branch"):
    inp = Input(shape=(seq_len, n_features), name=f"input_{name}")
    x   = Dense(embed_dim, activation='linear', name=f"proj_{name}")(inp)
    x   = LayerNormalization(epsilon=1e-6)(x)
    for _ in range(TRANSFORMER_LAYERS):
        x = transformer_block(x, embed_dim, N_HEADS, FF_DIM, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    return inp, x


# ─────────────────────────────────────────────
# DIRECTION MODEL  (single output — class_weight works)
# ─────────────────────────────────────────────
def build_model(m15_feat: int, h1_feat: int) -> Model:
    # H1 branch — macro trend context
    inp_h1,  xh = build_branch(SEQ_LEN_H1,  h1_feat,  embed_dim=32, name="h1")
    # M15 branch — entry signal
    inp_m15, xm = build_branch(SEQ_LEN_M15, m15_feat, embed_dim=48, name="m15")

    # H1 first — macro context anchors the merge
    merged = concatenate([xh, xm], name="merge")
    trunk  = Dense(64, activation='relu')(merged)
    trunk  = Dropout(0.3)(trunk)
    trunk  = Dense(32, activation='relu')(trunk)
    trunk  = Dropout(0.3)(trunk)
    out    = Dense(3, activation='softmax', name='direction')(trunk)

    model = Model(inputs=[inp_m15, inp_h1], outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# KELLY MODEL
# ─────────────────────────────────────────────
def build_kelly_model(m15_feat: int) -> Model:
    inp, x = build_branch(SEQ_LEN_M15, m15_feat, embed_dim=16, name="kelly")
    x   = Dense(16, activation='relu')(x)
    out = Dense(1,  activation='sigmoid', name='kelly')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


# ─────────────────────────────────────────────
# REGIME MODEL
# ─────────────────────────────────────────────
def build_regime_model(m15_feat: int) -> Model:
    inp, x = build_branch(SEQ_LEN_M15, m15_feat, embed_dim=16, name="regime")
    out    = Dense(1, activation='sigmoid', name='regime_out')(x)
    model  = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ─────────────────────────────────────────────
# KELLY TARGETS
# ─────────────────────────────────────────────
def make_kelly_targets(y_labels: np.ndarray,
                       window: int = 200) -> np.ndarray:
    kelly = np.zeros(len(y_labels), dtype=np.float32)
    for i in range(len(y_labels)):
        start       = max(0, i - window)
        segment     = y_labels[start: i + 1]
        non_neutral = segment[segment != 1]
        if len(non_neutral) == 0:
            continue
        win_rate = np.mean(non_neutral == 2)
        edge     = abs(win_rate - 0.5)
        fraction = edge / (1 - edge + 1e-9)
        kelly[i] = float(np.clip(fraction, 0.0, 1.0))
    return kelly


# ─────────────────────────────────────────────
# tf.data PIPELINE  (faster than raw numpy)
# ─────────────────────────────────────────────
def make_dataset(X_m15, X_h1, y,
                 batch_size: int = 128,
                 shuffle: bool = True,
                 sample_weights=None):
    inputs = {'input_m15': X_m15, 'input_h1': X_h1}
    if sample_weights is not None:
        ds = tf.data.Dataset.from_tensor_slices((inputs, y, sample_weights))
    else:
        ds = tf.data.Dataset.from_tensor_slices((inputs, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m15_csv",  required=True)
    parser.add_argument("--h1_csv",   required=True)
    parser.add_argument("--n_models", type=int, default=N_MODELS)
    args = parser.parse_args()

    # GPU check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"GPU: {gpus[0].name}")
    else:
        print("WARNING: No GPU found — training on CPU (will be slow)")

    # ── Load ──────────────────────────────────
    print("\n=== Loading Data ===")
    df_m15 = load_mt_csv(args.m15_csv)
    df_h1  = load_mt_csv(args.h1_csv)

    # ── Features ──────────────────────────────
    print("\n=== Generating Features ===")
    X_m15_df = add_features(df_m15)
    X_h1_df  = add_features(df_h1)
    n_feat   = X_m15_df.shape[1]
    print(f"  Features per timeframe: {n_feat}")

    # ── Labels ────────────────────────────────
    print("\n=== Generating Labels ===")
    labels        = make_labels(df_m15)
    regime_labels = make_regime_labels(df_m15)
    valid_times   = labels.dropna().index.sort_values()
    print(f"  Total valid label timestamps: {len(valid_times):,}")

    # ── Walk-forward folds ────────────────────
    print("\n=== Walk-Forward Folds ===")
    folds = make_walk_forward_folds(valid_times, N_FOLDS)

    # ── Scalers — fit on first 60% only ───────
    split_strict = (
        valid_times[int(len(valid_times) * 0.6)] - pd.Timedelta(minutes=1)
    )
    print(f"\n=== Fitting Scalers (up to {split_strict.date()}) ===")
    scaler_m15 = StandardScaler().fit(X_m15_df.loc[:split_strict])
    scaler_h1  = StandardScaler().fit(X_h1_df.loc[:split_strict])
    joblib.dump(scaler_m15, SCALER_M15_OUT)
    joblib.dump(scaler_h1,  SCALER_H1_OUT)
    print(f"  Scalers saved.")

    X_m15_sc = pd.DataFrame(scaler_m15.transform(X_m15_df),
                             index=X_m15_df.index, columns=X_m15_df.columns)
    X_h1_sc  = pd.DataFrame(scaler_h1.transform(X_h1_df),
                             index=X_h1_df.index,  columns=X_h1_df.columns)

    # ── Regime model ──────────────────────────
    print("\n=== Training Regime Model ===")
    X_m15_all, _, y_reg_all = create_sequences(
        X_m15_sc, X_h1_sc, regime_labels, valid_times
    )
    split_r      = int(len(y_reg_all) * 0.8)
    regime_model = build_regime_model(n_feat)
    regime_model.fit(
        X_m15_all[:split_r], y_reg_all[:split_r],
        validation_data=(X_m15_all[split_r:], y_reg_all[split_r:]),
        epochs=20, batch_size=128,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1
    )
    regime_model.save(REGIME_OUT)
    print(f"  Regime model saved -> {REGIME_OUT}")

    # ── Kelly model ───────────────────────────
    print("\n=== Training Kelly Model ===")
    X_m15_all, _, y_dir_all = create_sequences(
        X_m15_sc, X_h1_sc, labels, valid_times
    )
    kelly_all   = make_kelly_targets(y_dir_all)
    split_k     = int(len(y_dir_all) * 0.8)
    kelly_model = build_kelly_model(n_feat)
    kelly_model.fit(
        X_m15_all[:split_k], kelly_all[:split_k],
        validation_data=(X_m15_all[split_k:], kelly_all[split_k:]),
        epochs=20, batch_size=128,
        callbacks=[EarlyStopping(patience=4, restore_best_weights=True)],
        verbose=1
    )
    kelly_model.save(KELLY_OUT)
    print(f"  Kelly model saved -> {KELLY_OUT}")

    # ─────────────────────────────────────────
    # ENSEMBLE x WALK-FORWARD
    # ─────────────────────────────────────────
    print(f"\n=== Ensemble Training "
          f"({args.n_models} models x {len(folds)} folds) ===")

    fold_results = []

    for fold_idx, (train_times, test_times) in enumerate(folds):
        print(f"\n-- Fold {fold_idx + 1}/{len(folds)} --")

        X_m15_tr, X_h1_tr, y_tr = create_sequences(
            X_m15_sc, X_h1_sc, labels, train_times
        )
        X_m15_te, X_h1_te, y_te = create_sequences(
            X_m15_sc, X_h1_sc, labels, test_times
        )
        print(f"  train={len(y_tr):,}  test={len(y_te):,}")

        if len(y_tr) < 200 or len(y_te) < 50:
            print("  Too few samples — skipping fold")
            continue

        # Balanced class weights
        cw     = compute_class_weight('balanced',
                                      classes=np.unique(y_tr), y=y_tr)
        cw_dict = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw)}
        print(f"  Class weights: {cw_dict}")

        # Recency weights × class weights = combined sample weights
        rec_w  = make_recency_weights(train_times[:len(y_tr)])
        cls_w  = np.array([cw_dict[int(c)] for c in y_tr], dtype=np.float32)
        sw     = (rec_w * cls_w).astype(np.float32)

        # tf.data pipelines
        train_ds = make_dataset(X_m15_tr, X_h1_tr, y_tr,
                                batch_size=128, shuffle=True,
                                sample_weights=sw)
        val_ds   = make_dataset(X_m15_te, X_h1_te, y_te,
                                batch_size=128, shuffle=False)

        fold_model_paths = []

        for m_idx in range(args.n_models):
            seed = 42 + fold_idx * 100 + m_idx
            set_seed(seed)

            model_path = os.path.join(
                MODEL_DIR, f"model_fold{fold_idx+1}_m{m_idx+1}.keras"
            )
            print(f"\n  Model {m_idx+1}/{args.n_models}  seed={seed}")
            model = build_model(n_feat, n_feat)

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
                    model_path, save_best_only=True,
                    monitor='val_accuracy'
                )
            ]

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=60,
                callbacks=callbacks,
                verbose=1
            )

            best_acc = max(history.history['val_accuracy'])
            print(f"  Best val_accuracy: {best_acc:.4f}")
            fold_model_paths.append((model_path, best_acc))

        fold_results.append({
            'fold'  : fold_idx + 1,
            'models': fold_model_paths
        })

    # ── Summary ───────────────────────────────
    print("\n=======================================")
    print("  TRAINING COMPLETE - ENSEMBLE SUMMARY")
    print("=======================================")
    all_accs = []
    for fr in fold_results:
        for path, acc in fr['models']:
            all_accs.append(acc)
            print(f"  Fold {fr['fold']}  "
                  f"{os.path.basename(path):35s}  acc={acc:.4f}")
    if all_accs:
        print(f"\n  Mean val accuracy : {np.mean(all_accs):.4f}")
        print(f"  Best val accuracy : {max(all_accs):.4f}")

    print(f"\n  Ensemble  -> {MODEL_DIR}/")
    print(f"  Regime    -> {REGIME_OUT}")
    print(f"  Kelly     -> {KELLY_OUT}")
    print(f"  Scalers   -> {SCALER_M15_OUT}, {SCALER_H1_OUT}")


if __name__ == "__main__":
    main()

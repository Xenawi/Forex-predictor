import pandas as pd
import numpy as np

# ── Config ──
FUTURE_WINDOW = 12
ATR_MULT      = 0.9

def load_mt_csv(path):
    df = pd.read_csv(path, header=None)
    df.columns = ['date','time','open','high','low','close','volume']
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.drop_duplicates(subset=['datetime'], keep='last')
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna().set_index('datetime').sort_index()

def ATR(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    return pd.concat([hl,hc,lc], axis=1).max(axis=1).rolling(period).mean()

def check_labels(path):
    df     = load_mt_csv(path)
    atr    = ATR(df)
    labels = pd.Series(index=df.index, dtype=float)
    counts = {0:0, 1:0, 2:0, 'ambig':0}

    for i in range(len(df) - FUTURE_WINDOW):
        if pd.isna(atr.iloc[i]):
            continue
        upper        = df['close'].iloc[i] + atr.iloc[i] * ATR_MULT
        lower        = df['close'].iloc[i] - atr.iloc[i] * ATR_MULT
        future_slice = df.iloc[i+1 : i+1+FUTURE_WINDOW]
        hit_buy      = (future_slice['high'] >= upper).any()
        hit_sell     = (future_slice['low']  <= lower).any()

        if   hit_buy  and not hit_sell: labels.iloc[i] = 2.0; counts[2] += 1
        elif hit_sell and not hit_buy:  labels.iloc[i] = 0.0; counts[0] += 1
        elif not hit_buy and not hit_sell: labels.iloc[i] = 1.0; counts[1] += 1
        else: counts['ambig'] += 1

    total = counts[0] + counts[1] + counts[2]
    print(f"\nLabel Distribution:")
    print(f"  SELL    : {counts[0]:>8,}  ({100*counts[0]/max(total,1):.1f}%)")
    print(f"  NEUTRAL : {counts[1]:>8,}  ({100*counts[1]/max(total,1):.1f}%)")
    print(f"  BUY     : {counts[2]:>8,}  ({100*counts[2]/max(total,1):.1f}%)")
    print(f"  Ambig   : {counts['ambig']:>8,}  (discarded)")
    print(f"  Total   : {total:>8,}")

# ── Replace with your actual file path ──
check_labels("EURUSD_M15.csv")

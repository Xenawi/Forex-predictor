import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
from datetime import datetime, time as dtime
from tensorflow.keras.models import load_model
from ta.trend import ADXIndicator
import os

# ===================== CONFIG =====================

SYMBOL = "EURUSD"

SEQ_LEN = 30
CONFIRM_BARS = 2
BASE_THRESHOLD = 0.75
COOLDOWN_BARS = 3

# ===================== LOG FILES =====================

LOG_SIGNALS = "ai_signals_log.csv"
LOG_TRADES = "ai_trades_log.csv"
LOG_EQUITY = "ai_equity_log.csv"

# ===== RISK MANAGEMENT =====

RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.04
MAX_TRADES_PER_DAY = 6
MAX_SPREAD_PIPS = 2.5
MIN_ATR = 0.00008
MAX_DRAWDOWN = 0.12

USE_TRADING_SESSION = False

MODEL_PATH = "forex_model_M15_binary.keras"
SCALER_M1_PATH = "scaler_m1.pkl"
SCALER_M15_PATH = "scaler_m15.pkl"

# ===================== STATE =====================

signal_buffer = []
last_bar = None
cooldown = COOLDOWN_BARS
daily_trades = 0
daily_start_balance = None
peak_equity = None
current_day = None

# ===================== LOAD =====================

MODEL = load_model(MODEL_PATH)
SCALER_M1 = joblib.load(SCALER_M1_PATH)
SCALER_M15 = joblib.load(SCALER_M15_PATH)

# ===================== MT5 INIT =====================

if not mt5.initialize():
    raise RuntimeError("MT5 init failed")

mt5.symbol_select(SYMBOL, True)
print("✅ Institutional AI Bot started")

# ===================== HELPERS =====================

def append_csv(path, row_dict):
    df = pd.DataFrame([row_dict])
    file_exists = os.path.isfile(path)
    df.to_csv(path, mode='a', header=not file_exists, index=False)

def get_account():
    return mt5.account_info()

def get_balance():
    info = get_account()
    return info.balance if info else None

def get_equity():
    info = get_account()
    return info.equity if info else None

def has_open_position():
    pos = mt5.positions_get(symbol=SYMBOL)
    return pos is not None and len(pos) > 0

def get_spread_pips():
    tick = mt5.symbol_info_tick(SYMBOL)
    info = mt5.symbol_info(SYMBOL)
    if tick is None or info is None:
        return 999
    spread_points = (tick.ask - tick.bid) / info.point
    return spread_points / 10

def in_trading_session():
    if not USE_TRADING_SESSION:
        return True

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False

    server_time = datetime.fromtimestamp(tick.time).time()
    london = dtime(7, 0) <= server_time <= dtime(16, 0)
    newyork = dtime(13, 0) <= server_time <= dtime(21, 0)
    return london or newyork

def confirm_signal(direction):
    global signal_buffer

    signal_buffer.append(direction)
    if len(signal_buffer) > CONFIRM_BARS:
        signal_buffer.pop(0)

    if len(signal_buffer) < CONFIRM_BARS:
        return False

    return all(s == direction for s in signal_buffer)

def market_is_trending(feats_df):
    if feats_df is None or len(feats_df) == 0:
        return False

    if 'adx14' not in feats_df.columns:
        print("⚠️ ADX missing — blocking trade")
        return False

    return feats_df['adx14'].iloc[-1] > 14

# ===================== DATA =====================

def get_m1_data(n):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open','high','low','close','volume']]

def get_m15_data(n):
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, n)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    return df[['open','high','low','close','volume']]

# ===================== FEATURES =====================

def add_features(df):
    df = df.copy()

    df['close_ret'] = df['close'].pct_change().fillna(0)
    df['ema_dist'] = (df['close'] - df['close'].ewm(span=21).mean()) / df['close']
    df['hl_range'] = (df['high'] - df['low']) / df['close']
    df['ema_diff'] = df['close'].ewm(span=9).mean() - df['close'].ewm(span=21).mean()

    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df['rsi14'] = 100 - (100 / (1 + rs))

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)

    df['atr14'] = tr.rolling(14).mean()

    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = df['low'].diff().abs().clip(lower=0)

    atr = df['atr14']
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-9))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)) * 100

    df['adx14'] = dx.rolling(14).mean()

    df['momentum'] = df['close'] - df['close'].shift(5)

    df['ema_trend'] = (
        df['close'].ewm(span=50).mean() >
        df['close'].ewm(span=200).mean()
    ).astype(int)

    FEATURES = [
        'close_ret','ema_dist','hl_range','volume','ema_diff',
        'rsi14','atr14','adx14','momentum','ema_trend'
    ]

    return df[FEATURES].dropna()

# ===================== PREDICT =====================

def predict(df_m1, df_m15):

    feats_m1 = add_features(df_m1)
    feats_m15 = add_features(df_m15)

    if len(feats_m1) < SEQ_LEN or len(feats_m15) < SEQ_LEN:
        return None, None, None, None, None

    atr = feats_m1['atr14'].iloc[-1]

    X_m1 = SCALER_M1.transform(feats_m1)
    X_m15 = SCALER_M15.transform(feats_m15)

    X_m1_seq = X_m1[-SEQ_LEN:].reshape(1, SEQ_LEN, X_m1.shape[1])
    X_m15_seq = X_m15[-SEQ_LEN:].reshape(1, SEQ_LEN, X_m15.shape[1])

    prob_buy = float(MODEL.predict([X_m1_seq, X_m15_seq], verbose=0)[0][0])
    prob_sell = 1.0 - prob_buy

    if prob_buy >= BASE_THRESHOLD:
        direction = "BUY"
        confidence = prob_buy
    elif prob_buy <= (1 - BASE_THRESHOLD):
        direction = "SELL"
        confidence = prob_sell
    else:
        direction = None
        confidence = prob_buy

    return direction, confidence, atr, (prob_buy, prob_sell), feats_m1

# ===================== LOT SIZE =====================

def compute_lot(atr):
    balance = get_balance()
    if balance is None or atr <= 0:
        return 0.01

    risk_money = balance * RISK_PER_TRADE
    sl_distance = atr * 2.5
    pip_value = 10

    lot = risk_money / (sl_distance / 0.0001 * pip_value)
    return max(0.01, round(lot, 2))

# ===================== TRADE =====================

def open_trade(direction, atr):

    lot = compute_lot(atr)

    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return None, lot

    price = tick.ask if direction=="BUY" else tick.bid

    sl_dist = atr * 2.5
    tp_dist = sl_dist * 1.8

    sl = price - sl_dist if direction=="BUY" else price + sl_dist
    tp = price + tp_dist if direction=="BUY" else price - tp_dist

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 999,
        "comment": "AI_INSTITUTIONAL",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    return mt5.order_send(request), lot

# ===================== MAIN LOOP =====================

while True:
    try:

        if not in_trading_session():
            time.sleep(30)
            continue

        today = datetime.now().date()
        if current_day != today:
            current_day = today
            daily_trades = 0
            daily_start_balance = get_balance()

        equity = get_equity()
        if equity is None:
            time.sleep(2)
            continue

        if peak_equity is None:
            peak_equity = equity
        peak_equity = max(peak_equity, equity)

        if equity < peak_equity * (1 - MAX_DRAWDOWN):
            print("🛑 Max drawdown hit")
            time.sleep(60)
            continue

        if daily_start_balance and get_balance() < daily_start_balance * (1 - MAX_DAILY_LOSS):
            print("🛑 Daily loss limit hit")
            time.sleep(60)
            continue

        if daily_trades >= MAX_TRADES_PER_DAY:
            time.sleep(10)
            continue

        if get_spread_pips() > MAX_SPREAD_PIPS:
            print("⚠️ Spread too high")
            time.sleep(5)
            continue

        df_m1 = get_m1_data(SEQ_LEN + 100)
        df_m15 = get_m15_data(SEQ_LEN + 100)

        if df_m1 is None or df_m15 is None:
            time.sleep(2)
            continue

        now = df_m1.index[-1]

        if last_bar is None:
            last_bar = now
            print("🟡 First bar initialized")
            time.sleep(1)
            continue

        if now == last_bar:
            time.sleep(0.5)
            continue

        last_bar = now
        cooldown += 1

        direction, confidence, atr, probs, feats_m1 = predict(df_m1, df_m15)

        if probs is not None:
            prob_buy, prob_sell = probs
            print(f"{now} | BUY={prob_buy:.3f} | SELL={prob_sell:.3f}")
        else:
            print(f"{now} | No prediction")
            continue

        # print(f"{now} | BUY={prob_buy:.3f} | SELL={prob_sell:.3f}")

        append_csv(LOG_SIGNALS, {
            "time": now,
            "buy_conf": prob_buy,
            "sell_conf": prob_sell,
            "decision": direction
        })

        if direction is None:
            continue

        print(f"ATR Value: {atr}")
        if atr < MIN_ATR:
            print("⚠️ ATR too low")
            continue

        if not market_is_trending(feats_m1):
            print("❌ Blocked: market not trending")
            continue

        if not confirm_signal(direction):
            print("❌ Blocked: confirmation failed")
            continue

        if has_open_position():
            print("❌ Blocked: position already open")
            continue

        if cooldown < COOLDOWN_BARS:
            print(f"❌ Blocked: cooldown {cooldown}/{COOLDOWN_BARS}")
            continue

        result, lot = open_trade(direction, atr)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            cooldown = 0
            daily_trades += 1
            print("✅ Trade opened")

            append_csv(LOG_TRADES, {
                "time": now,
                "symbol": SYMBOL,
                "direction": direction,
                "volume": lot,
                "price": result.price,
                "confidence": confidence
            })

        time.sleep(1)

    except KeyboardInterrupt:
        print("🛑 Bot stopped")
        break
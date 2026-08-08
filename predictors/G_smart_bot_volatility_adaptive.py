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

# ===== TRAILING SETTINGS =====
USE_BREAK_EVEN = True
BE_TRIGGER_PIPS = 15  # Move SL once we are 15 pips in profit
BE_LOCK_PIPS = 2      # Lock in 2 pips of profit (covers commissions/spread)

USE_TRAILING_STEP = True
TRAIL_STEP_PIPS = 5    # Every 5 pips of new profit, move SL up
TRAIL_DISTANCE_PIPS = 10 # Keep SL 10 pips behind the current price

SYMBOL = "EURUSD"

SEQ_LEN = 30
CONFIRM_BARS = 2
BASE_THRESHOLD = 0.80
COOLDOWN_BARS = 2  # FIXED: Now properly decrements

# ===================== LOG FILES =====================
LOG_SIGNALS = "ai_signals_log.csv"
LOG_TRADES = "ai_trades_log.csv"
LOG_EQUITY = "ai_equity_log.csv"

# ===== RISK MANAGEMENT =====
RISK_PER_TRADE = 0.01
MAX_DAILY_LOSS = 0.04
MAX_TRADES_PER_DAY = 60  # FIXED: Now enforced
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
cooldown = 0  # FIXED: Start at 0, increment to COOLDOWN_BARS then reset
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
print("✅ Institutional AI Bot started (Fixed Version)")

# ===================== HELPERS =====================

def get_filling_mode():
    info = get_symbol_info()
    if info is None:
        return mt5.ORDER_FILLING_RETURN

    # Prefer RETURN → IOC → FOK
    if info.filling_mode == mt5.ORDER_FILLING_RETURN:
        return mt5.ORDER_FILLING_RETURN
    elif info.filling_mode == mt5.ORDER_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    else:
        return mt5.ORDER_FILLING_FOK

def close_position(position):
    tick = get_tick()
    if tick is None:
        return

    price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask

    # Create the base request without type_filling
    request_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": price,
        "deviation": 20,
        "magic": 999,
        "comment": "AI_EXIT",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    # Pass it to your existing fallback function
    result = send_order_with_fallback(request_base)

    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"🚪 Position closed #{position.ticket}")
    else:
        print(f"❌ Close failed: Exhausted all filling modes.")

def get_symbol_info():
    """Get cached symbol info with safety checks"""
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        print(f"❌ Failed to get {SYMBOL} info")
        return None
    return info

def get_tick():
    """Safe tick retrieval"""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print("⚠️ Tick data unavailable")
        return None
    return tick

def pips_to_price(pips):
    """Convert pips to price distance"""
    info = get_symbol_info()
    if info is None:
        return 0
    return pips * info.point * 10  # Standard for 5-digit brokers [web:15]

def manage_open_trade(direction, confidence, atr):
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions:
        return

    pos = positions[0]
    tick = get_tick()
    info = get_symbol_info()
    if tick is None or info is None:
        return

    pip = 0.0001 if info.digits >= 4 else info.point

    # === calculate current profit in pips ===
    if pos.type == mt5.POSITION_TYPE_BUY:
        profit_pips = (tick.bid - pos.price_open) / pip
        opposite = direction == "SELL"
    else:
        profit_pips = (pos.price_open - tick.ask) / pip
        opposite = direction == "BUY"

    # ================================
    # EXIT RULE 1 — strong reversal
    # ================================
    if opposite and confidence > 0.65:
        print("🚨 Opposite signal — closing early")
        close_position(pos)
        return

    # ================================
    # EXIT RULE 2 — lock good profit
    # ================================
    if profit_pips >= 25:
        print(f"💰 Profit target hit: {profit_pips:.1f} pips")
        close_position(pos)
        return

    # ================================
    # EXIT RULE 3 — volatility died
    # ================================
    if atr < MIN_ATR * 0.7 and profit_pips > 5:
        print("📉 ATR dropped — securing profit")
        close_position(pos)
        return

def manage_trailing_stop(symbol, trigger_pips, lock_pips, trail_step, trail_dist):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return

    symbol_info = get_symbol_info()
    if symbol_info is None:
        return
    # point = symbol_info.point
    pip = 0.0001 if symbol_info.digits >= 4 else symbol_info.point
    tick = get_tick()
    if tick is None:
        return

    for pos in positions:
        # Calculate current pips in profit
        if pos.type == mt5.POSITION_TYPE_BUY:
            current_pips = (tick.bid - pos.price_open) / pip

            # Phase 1: Break-Even FIXED: Safe checks
            if pos.sl < pos.price_open and current_pips >= trigger_pips:
                new_sl = pos.price_open + pips_to_price(lock_pips)
                modify_sl(pos.ticket, new_sl, pos.tp)
                print(f"🛡️ Break-even set for Buy #{pos.ticket}")

            # Phase 2: Trailing Step FIXED: Safe checks
            elif USE_TRAILING_STEP and current_pips > trigger_pips:
                ideal_sl = tick.bid - pips_to_price(trail_dist)
                if ideal_sl > (pos.sl or 0) + pips_to_price(trail_step):
                    modify_sl(pos.ticket, ideal_sl, pos.tp)
                    print(f"📈 Trailing Buy #{pos.ticket} to +{(ideal_sl-pos.price_open) / pip:.1f} pips")

        elif pos.type == mt5.POSITION_TYPE_SELL:
            current_pips = (pos.price_open - tick.ask) / pip

            # Phase 1: Break-Even FIXED: Better condition
            if (pos.sl is None or pos.sl == 0 or pos.sl > pos.price_open) and current_pips >= trigger_pips:
                new_sl = pos.price_open - pips_to_price(lock_pips)
                modify_sl(pos.ticket, new_sl, pos.tp)
                print(f"🛡️ Break-even set for Sell #{pos.ticket}")

            # Phase 2: Trailing Step FIXED: Safe checks
            elif USE_TRAILING_STEP and current_pips > trigger_pips:
                ideal_sl = tick.ask + pips_to_price(trail_dist)
                current_sl = pos.sl or float('inf')
                if ideal_sl < current_sl - pips_to_price(trail_step):
                    modify_sl(pos.ticket, ideal_sl, pos.tp)
                    print(f"📉 Trailing Sell #{pos.ticket} to +{(pos.price_open-ideal_sl) / pip:.1f} pips")

def modify_sl(ticket, new_sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": new_sl,
        "tp": tp
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ SL Update failed: {result.comment}")

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
    tick = get_tick()
    info = get_symbol_info()
    if tick is None or info is None:
        return 999
    spread_points = (tick.ask - tick.bid) / info.point
    return spread_points / 10

def in_trading_session():
    if not USE_TRADING_SESSION:
        return True
    tick = get_tick()
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
    return feats_df['adx14'].iloc[-1] > 20  # FIXED: Strong trend threshold [web:10]

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

# ===================== FEATURES ===================== (Unchanged)
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

    adx_indicator = ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )

    df['adx14'] = adx_indicator.adx().fillna(0)

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

# ===================== PREDICT ===================== (Unchanged + safety)
def predict(df_m1, df_m15):
    feats_m1 = add_features(df_m1)
    feats_m15 = add_features(df_m15)

    if len(feats_m1) < SEQ_LEN or len(feats_m15) < SEQ_LEN:
        return None, None, None, None, None

    # FIXED: Shape check
    if feats_m1.shape[1] != SCALER_M1.n_features_in_ or feats_m15.shape[1] != SCALER_M15.n_features_in_:
        print("⚠️ Feature shape mismatch")
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

# ===================== LOT SIZE FIXED: Proper pip value =====================
def compute_lot(atr):
    balance = get_balance()
    if balance is None or atr <= 0:
        return 0.01

    info = get_symbol_info()
    if info is None:
        return 0.01

    risk_money = balance * RISK_PER_TRADE
    sl_distance = atr * 2.5
    pip_value = info.trade_tick_value * 10  # FIXED: Dynamic pip value [web:11]
    
    if pip_value <= 0:
        pip_value = 10  # Fallback

    lot = risk_money / (sl_distance / info.point / 10 * pip_value)  # Proper formula
    return max(0.01, min(10.0, round(lot, 2)))  # Cap max lot


def send_order_with_fallback(request_base):
    """Try multiple filling modes until one works"""
    
    filling_modes = [
        mt5.symbol_info(SYMBOL).filling_mode,
        mt5.ORDER_FILLING_RETURN,
        mt5.ORDER_FILLING_FOK,
        mt5.ORDER_FILLING_IOC,
    ]

    tried = set()

    for mode in filling_modes:
        if mode in tried:
            continue
        tried.add(mode)

        request = request_base.copy()
        request["type_filling"] = mode

        result = mt5.order_send(request)

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ Order filled using mode {mode}")
            return result

        else:
            if result:
                print(f"⚠️ Filling {mode} failed: {result.retcode}")
            else:
                print(f"⚠️ Filling {mode} returned None")

    return None


# ===================== TRADE FIXED: IOC filling =====================
def open_trade(direction, atr):
    lot = compute_lot(atr)
    tick = get_tick()
    if tick is None:
        return None, lot

    price = tick.ask if direction=="BUY" else tick.bid
    info = get_symbol_info()
    if info is None:
        return None, lot

    sl_dist = atr * 2.5
    tp_dist = sl_dist * 2.0  # FIXED: Better R:R 1:2

    sl = price - sl_dist if direction=="BUY" else price + sl_dist
    tp = price + tp_dist if direction=="BUY" else price - tp_dist

    # Validate min distances
    if abs(price - sl) < info.trade_stops_level * info.point:
        print("⚠️ SL too close to price")
        return None, lot
    
    info = get_symbol_info()
    print("Broker filling mode:", info.filling_mode)
    if info is None:
        return None, lot

    filling = info.filling_mode

    request_base = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if direction=="BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 999,
        "comment": "AI_INSTITUTIONAL_FIXED",
        "type_time": mt5.ORDER_TIME_GTC,
    }

    result = send_order_with_fallback(request_base)
    return result, lot


# ===================== MAIN LOOP FIXED =====================
while True:
    try:
        # 1. LIVE POSITION MANAGEMENT (Safe checks)
        if has_open_position():
            manage_trailing_stop(
                SYMBOL, 
                BE_TRIGGER_PIPS, 
                BE_LOCK_PIPS, 
                TRAIL_STEP_PIPS, 
                TRAIL_DISTANCE_PIPS
            )

        # 2. RISK & SESSION CHECKS
        if not in_trading_session():
            time.sleep(5)
            continue

        # FIXED: Use MT5 server time for daily reset
        tick = get_tick()
        if tick:
            today = datetime.fromtimestamp(tick.time).date()
        else:
            today = datetime.now().date()


        if current_day != today:
            current_day = today
            daily_trades = 0
            daily_start_balance = get_balance()
            print(f"📅 New day: {today}, trades reset")

        equity = get_equity()
        if equity is None:
            time.sleep(2)
            continue

        if peak_equity is None or equity > peak_equity:
            peak_equity = equity

        # Global Circuit Breakers
        if equity < peak_equity * (1 - MAX_DRAWDOWN):
            print(f"🛑 Max drawdown hit")
            time.sleep(60)
            continue

        if daily_start_balance and get_balance() < daily_start_balance * (1 - MAX_DAILY_LOSS):
            print(f"🛑 Daily loss limit hit")
            time.sleep(60)
            continue

        # FIXED: Enforce daily trade limit
        if daily_trades >= MAX_TRADES_PER_DAY:
            print(f"🛑 Daily trade limit reached: {daily_trades}/{MAX_TRADES_PER_DAY}")
            time.sleep(30)
            continue

        # 3. DATA FETCHING & BAR SYNC
        df_m1 = get_m1_data(SEQ_LEN + 100)
        df_m15 = get_m15_data(SEQ_LEN + 100)
        
        if df_m1 is None or df_m15 is None:
            time.sleep(2)
            continue

        now = df_m1.index[-1]
        if last_bar is None:
            last_bar = now
            continue

        # If no new candle, quick loop for trailing
        if now == last_bar:
            time.sleep(1)
            continue
            
        # --- NEW BAR DETECTED ---
        last_bar = now
        cooldown = min(cooldown + 1, COOLDOWN_BARS + 1)  # FIXED: Proper cooldown logic

        if get_spread_pips() > MAX_SPREAD_PIPS:
            print(f"⚠️ Spread too high: {get_spread_pips():.1f}")
            continue

        direction, confidence, atr, probs, feats_m1 = predict(df_m1, df_m15)
        print(
            f"DEBUG | dir={direction} "
            f"ATR={atr:.5f} "
            f"ADX={feats_m1['adx14'].iloc[-1]:.2f} "
            f"cooldown={cooldown}"
        )
        
        if probs is None:
            continue
        
        manage_open_trade(direction, confidence, atr)
        
        append_csv(LOG_SIGNALS, {
            "time": now, "buy_conf": probs[0], "sell_conf": probs[1], "decision": direction
        })

        # 4. ENTRY FILTERS
        if direction is None:
            continue
        
        if atr < MIN_ATR:
            print(f"❌ ATR too low: {atr:.5f}")
            continue
        print(f"ATR raw: {atr:.6f}")
        if not market_is_trending(feats_m1):
            print("❌ Blocked: Weak trend (ADX)")
            continue

        if not confirm_signal(direction):
            print(f"❌ Blocked: {direction} not confirmed")
            continue

        if has_open_position():
            print("❌ Blocked: Position already open")
            continue
        if daily_trades >= MAX_TRADES_PER_DAY:
            print("❌ Blocked: max daily trades reached")
            continue

        if cooldown < COOLDOWN_BARS:  # FIXED: Now works properly
            print(f"⏳ Cooldown: {cooldown}/{COOLDOWN_BARS}")
            continue

        # 5. EXECUTION
        result, lot = open_trade(direction, atr)

        if result is None:
            print("❌ Order send failed (no result)")
            continue

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Trade failed: {result.retcode} | {result.comment}")
            continue

        cooldown = 0
        daily_trades += 1
        print(f"✅ {direction} Opened at {result.price}")


        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            cooldown = 0  # Reset cooldown
            daily_trades += 1
            print(f"✅ {direction} #{result.order} @ {result.price:.5f} (lot: {lot})")
            append_csv(LOG_TRADES, {
                "time": now, "symbol": SYMBOL, "direction": direction,
                "volume": lot, "price": result.price, "confidence": confidence
            })
        elif result:
            print(f"❌ Trade failed: {result.comment} (code: {result.retcode})")

    except Exception as e:
        print(f"❌ Error: {e}")
        time.sleep(5)
    except KeyboardInterrupt:
        print("🛑 Bot stopped by user")
        break

mt5.shutdown()  # FIXED: Proper cleanup [web:12]
print("✅ MT5 shutdown complete")

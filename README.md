# Forex AI Trading Project

## Overview

This repository contains code for training, validating, and running AI-based Forex trading models using MetaTrader 5 CSV data.

It supports:
- multi-timeframe feature generation from M1, M15, and H1 data
- ATR-based label creation for directional moves
- transformer and CNN/LSTM model architectures
- walk-forward validation with purge + embargo
- ensemble inference with multiple saved models
- auxiliary regime detection and Kelly position sizing
- live MT5 trading with risk management and trade logging

## Folder structure

- `training-files/`
  - Market data CSVs exported from MetaTrader 5
  - `EURUSD_H1.csv`, `EURUSD_M15.csv`, `FILE_DUMP_EURUSD_M1.csv`

- `ensemble_models/`
  - Saved model weights created by ensemble training scripts
  - Example: `model_fold1_m1.keras`

- `trained-models/`
  - Saved auxiliary models used by inference
  - Example: `kelly_model.keras`, `regime_model.keras`

- `predictors/`
  - `G_smart_bot_volatility_adaptive.py` — a live MT5 inference bot with volatility-adaptive trade rules

- `trainers/`
  - `G_train_ai_trad_M15.py` — original M15 training script for binary classification

## Key scripts

### `train_ai_trad_M15.py`

Trains a binary M15 model using M1 and M15 features. It saves:
- `forex_model_M15_binary.keras`
- `scaler_m1.pkl`
- `scaler_m15.pkl`

This model is used by the live bot in `smart_bot_volatility_adaptive.py`.

### `train_forex_v2.py`

A full ensemble training pipeline with:
- M1 + M15 + H1 transformer branches
- 3-class labels: `SELL`, `NEUTRAL`, `BUY`
- walk-forward validation and sample purge/embargo
- ensemble model generation
- auxiliary regime model and Kelly sizing model
- session and order-flow proxy features

Outputs:
- ensemble weights in `ensemble_models/`
- `regime_model.keras`
- `kelly_model.keras`
- `scaler_m15.pkl`
- `scaler_h1.pkl`

### `train_forex_v3.py`

A simplified two-timeframe ensemble pipeline focused on H1 trend context and M15 entry signals. It trains:
- ensembles of M15+H1 models
- a regime classification model
- a Kelly position-sizing model

It also saves ensemble weights, scalers, and auxiliary models.

### `inference.py`

Provides `ForexPredictor`, which loads:
- ensemble models from `ensemble_models/`
- `regime_model.keras`
- `kelly_model.keras`
- `scaler_m15.pkl`
- `scaler_h1.pkl`

The predictor returns:
- `label` (`BUY`, `SELL`, or `NO_TRADE`)
- direction probabilities
- `kelly_fraction`
- `regime` (`TRENDING` or `RANGING`)
- `confidence` (`HIGH` or `LOW`)

### `smart_bot_volatility_adaptive.py`

A live MetaTrader 5 trading bot that:
- connects to MT5 and selects `EURUSD`
- fetches live M1 and M15 price bars
- computes technical features (ATR, RSI, ADX, EMA, momentum)
- predicts direction using a binary M15 model
- confirms signals across bars
- enforces risk controls and volume sizing
- logs signals, trades, and equity to CSV files

## How it works

1. Load MT5 CSV market data.
2. Generate technical features on each timeframe.
3. Create target labels using ATR-based future move detection.
4. Align sequences across timeframes for model input.
5. Train models with walk-forward validation and class weighting.
6. Save models, scalers, and auxiliary predictors.
7. Run inference with an ensemble and auxiliary regime/Kelly models.
8. Optionally run the live MT5 bot for automated trading.

## Usage examples

Train the v3 ensemble:

```bash
python train_forex_v3.py --m15_csv training-files/EURUSD_M15.csv --h1_csv training-files/EURUSD_H1.csv
```

Train the binary M15 model:

```bash
python train_ai_trad_M15.py --m1_csv training-files/FILE_DUMP_EURUSD_M1.csv --m15_csv training-files/EURUSD_M15.csv
```

Run the inference module:

```bash
python inference.py
```

Run the live MT5 bot:

```bash
python smart_bot_volatility_adaptive.py
```

## Notes

- Update paths if saved model files are stored outside the working folder.
- Ensure required packages such as `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `joblib`, and `MetaTrader5` are installed.
- The live bot is written for `EURUSD` and uses ATR-based stops and profit management.

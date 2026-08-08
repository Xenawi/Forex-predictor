"""
inference_v3.py
===============
Live inference for the M15+H1 ensemble.

Usage in your MT5 bot:
    predictor = ForexPredictor()
    signal = predictor.predict(seq_m15, seq_h1)
    # Returns: {'label': 'BUY', 'p_buy': 0.67, 'kelly_fraction': 0.43,
    #           'regime': 'TRENDING', 'confidence': 'HIGH'}
"""

import os
import glob
import numpy as np
import joblib
import tensorflow as tf

MIN_CONFIDENCE   = 0.55
MIN_KELLY        = 0.10
REGIME_THRESHOLD = 0.5


class ForexPredictor:
    def __init__(self,
                 model_dir:       str = "ensemble_models",
                 regime_path:     str = "regime_model.keras",
                 kelly_path:      str = "kelly_model.keras",
                 scaler_m15_path: str = "scaler_m15.pkl",
                 scaler_h1_path:  str = "scaler_h1.pkl"):

        model_paths = sorted(glob.glob(os.path.join(model_dir, "*.keras")))
        assert model_paths, f"No models found in {model_dir}"
        self.models       = [tf.keras.models.load_model(p) for p in model_paths]
        self.regime_model = tf.keras.models.load_model(regime_path)
        self.kelly_model  = tf.keras.models.load_model(kelly_path)
        self.scaler_m15   = joblib.load(scaler_m15_path)
        self.scaler_h1    = joblib.load(scaler_h1_path)
        print(f"Loaded {len(self.models)} ensemble models + regime + kelly")

    def _scale(self, seq: np.ndarray, scaler) -> np.ndarray:
        """Scale (seq_len, n_feat) -> (1, seq_len, n_feat)"""
        return scaler.transform(seq)[np.newaxis, ...]

    def predict(self, seq_m15: np.ndarray, seq_h1: np.ndarray) -> dict:
        """
        Parameters
        ----------
        seq_m15 : np.ndarray  shape (SEQ_LEN_M15, n_features)
        seq_h1  : np.ndarray  shape (SEQ_LEN_H1,  n_features)

        Returns
        -------
        dict:
            label          - 'BUY' | 'SELL' | 'NO_TRADE'
            p_sell         - probability of SELL
            p_neutral      - probability of NEUTRAL
            p_buy          - probability of BUY
            kelly_fraction - suggested position size [0, 1]
            regime         - 'TRENDING' | 'RANGING'
            confidence     - 'HIGH' | 'LOW'
        """
        X_m15 = self._scale(seq_m15, self.scaler_m15)
        X_h1  = self._scale(seq_h1,  self.scaler_h1)

        # Average direction probabilities across ensemble
        probs = np.mean([
            model.predict([X_m15, X_h1], verbose=0)[0]
            for model in self.models
        ], axis=0)
        p_sell, p_neutral, p_buy = probs

        # Kelly and regime from M15 only
        kelly_frac  = float(self.kelly_model.predict(X_m15, verbose=0)[0, 0])
        regime_prob = float(self.regime_model.predict(X_m15, verbose=0)[0, 0])
        regime      = "TRENDING" if regime_prob >= REGIME_THRESHOLD else "RANGING"

        # Decision
        best_class = int(np.argmax(probs))
        best_prob  = float(probs[best_class])
        label      = {0: 'SELL', 1: 'NEUTRAL', 2: 'BUY'}[best_class]
        confidence = 'HIGH' if best_prob >= MIN_CONFIDENCE else 'LOW'

        # Suppress low confidence, low kelly, or neutral signals
        if confidence == 'LOW' or kelly_frac < MIN_KELLY or label == 'NEUTRAL':
            label = 'NO_TRADE'

        return {
            'label'          : label,
            'p_sell'         : round(float(p_sell),    4),
            'p_neutral'      : round(float(p_neutral), 4),
            'p_buy'          : round(float(p_buy),     4),
            'kelly_fraction' : round(kelly_frac,       4),
            'regime'         : regime,
            'confidence'     : confidence,
        }


if __name__ == "__main__":
    # Smoke test with random data
    SEQ_LEN_M15 = 48
    SEQ_LEN_H1  = 24
    N_FEATURES  = 13

    predictor = ForexPredictor()

    fake_m15 = np.random.randn(SEQ_LEN_M15, N_FEATURES).astype(np.float32)
    fake_h1  = np.random.randn(SEQ_LEN_H1,  N_FEATURES).astype(np.float32)

    result = predictor.predict(fake_m15, fake_h1)
    print("\nPrediction:")
    for k, v in result.items():
        print(f"  {k:18s}: {v}")

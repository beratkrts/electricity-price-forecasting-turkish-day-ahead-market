"""
[EXPERIMENTAL - Canlı pipeline'da kullanılmıyor]
Model routing modülü. Rejim tespitine göre farklı modeller arasında geçiş yapar.
Deneysel sonuçlar bu yaklaşımın LightGBM tek başına çalıştırmaktan daha iyi
sonuç vermediğini göstermiştir.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


class ModelRouter:
    """
    Piyasa rejimine (NORMAL, CRASH, VOLATILE) göre EPNet ve LightGBM tahminlerini yönlendiren/birleştiren sınıf.
    """
    def __init__(self, epnet_model: Any = None, lgbm_model: Any = None, regime_detector: Any = None, custom_weights: Optional[Dict] = None):
        self.epnet_model = epnet_model
        self.lgbm_model = lgbm_model
        self.regime_detector = regime_detector
        
        self.weights = custom_weights or {
            'NORMAL':   {'epnet': 1.0, 'lgbm': 0.0},
            'CRASH':    {'epnet': 0.0, 'lgbm': 1.0},
            'VOLATILE': {'epnet': 0.4, 'lgbm': 0.6},
        }

    def predict(self, df_features: pd.DataFrame, signals: Optional[Dict[str, float]] = None, epnet_preds: Optional[np.ndarray] = None, lgbm_preds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Rejimi tespit eder ve ağırlıklı tahmini hesaplar.
        """
        if signals is None and self.regime_detector is not None:
            signals = self.regime_detector.compute_signals(df_features)

        regime = 'NORMAL'
        if self.regime_detector is not None and signals is not None:
            regime = self.regime_detector.detect_regime(signals)

        # EPNet Tahmini
        if epnet_preds is None and self.epnet_model is not None:
            epnet_preds = self.epnet_model.predict(df_features)
        elif epnet_preds is None:
            epnet_preds = np.zeros(len(df_features))

        # LightGBM Tahmini
        if lgbm_preds is None and self.lgbm_model is not None:
            lgbm_preds = self.lgbm_model.predict(df_features)
        elif lgbm_preds is None:
            lgbm_preds = np.zeros(len(df_features))

        # Rejim ağırlıkları
        w = self.weights.get(regime, {'epnet': 0.5, 'lgbm': 0.5})
        final_preds = w['epnet'] * np.array(epnet_preds) + w['lgbm'] * np.array(lgbm_preds)

        return {
            'predictions': final_preds,
            'regime': regime,
            'epnet_preds': epnet_preds,
            'lgbm_preds': lgbm_preds,
            'weights': w,
            'signals': signals
        }

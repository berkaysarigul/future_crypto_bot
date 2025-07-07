import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("RegimeSwitcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RegimeSwitcher:
    """
    Production-level regime detection & switching.
    - Trend/range/volatile regime tespiti
    - Parametre switch, logging, exception handling
    - Outlier ve edge-case yönetimi
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vol_window = self.config.get("vol_window", 30)
        self.trend_window = self.config.get("trend_window", 30)
        self.trend_thresh = self.config.get("trend_thresh", 0.01)
        self.vol_thresh = self.config.get("vol_thresh", 0.02)
        self.current_regime = "trend"
        self.history = []

    def detect(self, price_series: np.ndarray) -> str:
        try:
            if len(price_series) < max(self.vol_window, self.trend_window):
                logger.warning("Yetersiz fiyat serisi, önceki regime döndürülüyor.")
                return self.current_regime
            # Trend ölçümü
            trend = (price_series[-1] - price_series[-self.trend_window]) / price_series[-self.trend_window]
            # Volatilite ölçümü
            vol = np.std(np.diff(price_series[-self.vol_window:])) / (np.mean(price_series[-self.vol_window:]) + 1e-8)
            # Outlier/edge-case kontrolü
            if np.isnan(trend) or np.isnan(vol) or np.isinf(trend) or np.isinf(vol):
                logger.error(f"Regime detect: NaN/Inf tespit edildi (trend={trend}, vol={vol})")
                return self.current_regime
            if abs(trend) > self.trend_thresh and vol < self.vol_thresh:
                regime = "trend"
            elif vol > self.vol_thresh:
                regime = "volatile"
            else:
                regime = "range"
            if regime != self.current_regime:
                logger.info(f"Regime değişti: {self.current_regime} → {regime} (trend={trend:.4f}, vol={vol:.4f})")
            self.current_regime = regime
            self.history.append({"trend": trend, "vol": vol, "regime": regime})
            return regime
        except Exception as e:
            logger.error(f"Regime detect hatası: {e}")
            return self.current_regime

    def switch(self, regime: str):
        logger.info(f"Manuel regime değişimi: {self.current_regime} → {regime}")
        self.current_regime = regime

    def get_context(self) -> np.ndarray:
        regimes = ["trend", "range", "volatile", "unknown"]
        context = np.zeros(len(regimes), dtype=np.float32)
        idx = regimes.index(self.current_regime) if self.current_regime in regimes else -1
        if idx >= 0:
            context[idx] = 1.0
        return context 
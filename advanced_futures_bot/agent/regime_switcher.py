from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

class RegimeSwitcher:
    """
    Trend/range regime detection ve otomatik parametre switch.
    - Fiyat serisi ve volatiliteye dayalı basit regime tespiti
    - RL agent için regime context üretir
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.vol_window = self.config.get("vol_window", 30)
        self.trend_window = self.config.get("trend_window", 30)
        self.trend_thresh = self.config.get("trend_thresh", 0.01)
        self.vol_thresh = self.config.get("vol_thresh", 0.02)
        self.current_regime = "trend"

    def detect(self, price_series: np.ndarray) -> str:
        """
        Basit regime tespiti: trend/range/volatile
        """
        if len(price_series) < max(self.vol_window, self.trend_window):
            return self.current_regime
        # Trend ölçümü (ör: son trend_window'da fiyat değişimi)
        trend = (price_series[-1] - price_series[-self.trend_window]) / price_series[-self.trend_window]
        # Volatilite ölçümü (ör: son vol_window'da std)
        vol = np.std(np.diff(price_series[-self.vol_window:])) / np.mean(price_series[-self.vol_window:])
        if abs(trend) > self.trend_thresh and vol < self.vol_thresh:
            self.current_regime = "trend"
        elif vol > self.vol_thresh:
            self.current_regime = "volatile"
        else:
            self.current_regime = "range"
        return self.current_regime

    def switch(self, regime: str):
        """Manuel regime değişimi (opsiyonel)."""
        self.current_regime = regime

    def get_context(self) -> np.ndarray:
        """
        RL agent için one-hot regime context (trend/range/volatile)
        """
        regimes = ["trend", "range", "volatile", "unknown"]
        context = np.zeros(len(regimes), dtype=np.float32)
        idx = regimes.index(self.current_regime) if self.current_regime in regimes else -1
        if idx >= 0:
            context[idx] = 1.0
        return context 
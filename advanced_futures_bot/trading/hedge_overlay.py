import logging
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("HedgeOverlay")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class HedgeOverlay:
    """
    Production-level hedge pair strategy logic.
    - Co-integration analysis, spread trading, perpetual premium
    - Production-level sinyal yönetimi, logging, exception handling
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("hedge_enabled", False)
        self.pairs = self.config.get("hedge_pairs", ["BTCUSDT", "ETHUSDT"])
        self.lookback = self.config.get("hedge_lookback", 100)
        self.z_threshold = self.config.get("z_threshold", 2.0)
        self.spread_threshold = self.config.get("spread_threshold", 0.01)
        self.min_correlation = self.config.get("min_correlation", 0.7)

    def calculate_cointegration(self, price1: np.ndarray, price2: np.ndarray) -> Dict[str, Any]:
        try:
            if len(price1) != len(price2) or len(price1) < self.lookback:
                logger.warning("Yetersiz veri için co-integration hesaplanamıyor")
                return {"cointegrated": False, "beta": 0.0, "residuals": None}
            slope, intercept, r_value, p_value, std_err = stats.linregress(price1, price2)
            residuals = price2 - (slope * price1 + intercept)
            # Basit stationarity check (variance-based)
            is_stationary = np.std(residuals) < np.std(price2) * 0.1
            correlation = abs(r_value)
            if correlation < self.min_correlation:
                logger.warning(f"Düşük korelasyon: {correlation:.4f}")
                is_stationary = False
            logger.info(f"Co-integration: {is_stationary}, beta={slope:.4f}, corr={correlation:.4f}")
            return {
                "cointegrated": is_stationary,
                "beta": slope,
                "intercept": intercept,
                "residuals": residuals,
                "r_squared": r_value ** 2,
                "correlation": correlation
            }
        except Exception as e:
            logger.error(f"Co-integration hesaplama hatası: {e}")
            return {"cointegrated": False, "beta": 0.0, "residuals": None}

    def calculate_spread(self, price1: float, price2: float, beta: float, intercept: float) -> float:
        try:
            expected_price2 = beta * price1 + intercept
            spread = (price2 - expected_price2) / max(expected_price2, 1e-4)
            return spread
        except Exception as e:
            logger.error(f"Spread hesaplama hatası: {e}")
            return 0.0

    def calculate_perpetual_premium(self, spot_price: float, futures_price: float) -> float:
        try:
            premium = (futures_price - spot_price) / max(spot_price, 1e-4)
            return premium
        except Exception as e:
            logger.error(f"Perpetual premium hesaplama hatası: {e}")
            return 0.0

    def generate_hedge_signals(self,
                              prices: Dict[str, np.ndarray],
                              current_prices: Dict[str, float]) -> Dict[str, Any]:
        try:
            if not self.enabled or len(self.pairs) < 2:
                return {"signals": [], "positions": {}}
            signals = []
            positions = {}
            price1 = prices.get(self.pairs[0], [])
            price2 = prices.get(self.pairs[1], [])
            if len(price1) > self.lookback and len(price2) > self.lookback:
                coint_result = self.calculate_cointegration(price1[-self.lookback:], price2[-self.lookback:])
                if coint_result["cointegrated"]:
                    current_price1 = current_prices.get(self.pairs[0], 0)
                    current_price2 = current_prices.get(self.pairs[1], 0)
                    if current_price1 > 0 and current_price2 > 0:
                        spread = self.calculate_spread(
                            current_price1, current_price2,
                            coint_result["beta"], coint_result["intercept"]
                        )
                        residuals = coint_result["residuals"]
                        z_score = (residuals[-1] - np.mean(residuals)) / max(np.std(residuals), 1e-4)
                        if abs(z_score) > self.z_threshold:
                            if z_score > 0:
                                signals.append({
                                    "type": "spread_short",
                                    "pair1": self.pairs[0],
                                    "pair2": self.pairs[1],
                                    "z_score": z_score,
                                    "spread": spread
                                })
                                logger.info(f"Hedge sinyali: spread_short, z_score={z_score:.4f}, spread={spread:.4f}")
                            else:
                                signals.append({
                                    "type": "spread_long",
                                    "pair1": self.pairs[0],
                                    "pair2": self.pairs[1],
                                    "z_score": z_score,
                                    "spread": spread
                                })
                                logger.info(f"Hedge sinyali: spread_long, z_score={z_score:.4f}, spread={spread:.4f}")
            return {"signals": signals, "positions": positions}
        except Exception as e:
            logger.error(f"Hedge sinyal üretme hatası: {e}")
            return {"signals": [], "positions": {}}

    def apply(self, prices: Dict[str, np.ndarray], current_prices: Dict[str, float]) -> Dict[str, Any]:
        try:
            if not self.enabled:
                return {"enabled": False}
            hedge_result = self.generate_hedge_signals(prices, current_prices)
            return {
                "enabled": True,
                "signals": hedge_result["signals"],
                "positions": hedge_result["positions"],
                "pairs": self.pairs
            }
        except Exception as e:
            logger.error(f"Hedge overlay uygulama hatası: {e}")
            return {"enabled": False, "error": str(e)} 
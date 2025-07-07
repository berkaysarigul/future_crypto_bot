from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

class HedgeOverlay:
    """
    Hedge pair strategy logic.
    - Co-integration analysis
    - Spread trading
    - Perpetual premium
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = self.config.get("hedge_enabled", False)
        self.pairs = self.config.get("hedge_pairs", ["BTCUSDT", "ETHUSDT"])
        self.lookback = self.config.get("hedge_lookback", 100)
        self.z_threshold = self.config.get("z_threshold", 2.0)
        self.spread_threshold = self.config.get("spread_threshold", 0.01)

    def calculate_cointegration(self, price1: np.ndarray, price2: np.ndarray) -> Dict[str, Any]:
        """
        Co-integration analizi.
        """
        if len(price1) != len(price2):
            return {"cointegrated": False, "beta": 0.0, "residuals": None}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(price1, price2)
        
        # Residuals
        residuals = price2 - (slope * price1 + intercept)
        
        # ADF test for stationarity
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(residuals)
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05
        except ImportError:
            # Fallback: simple variance check
            is_stationary = np.std(residuals) < np.std(price2) * 0.1
        
        return {
            "cointegrated": is_stationary,
            "beta": slope,
            "intercept": intercept,
            "residuals": residuals,
            "r_squared": r_value ** 2
        }

    def calculate_spread(self, price1: float, price2: float, beta: float, intercept: float) -> float:
        """
        Spread hesaplama.
        """
        expected_price2 = beta * price1 + intercept
        spread = (price2 - expected_price2) / expected_price2
        return spread

    def calculate_perpetual_premium(self, spot_price: float, futures_price: float) -> float:
        """
        Perpetual premium hesaplama.
        """
        premium = (futures_price - spot_price) / spot_price
        return premium

    def generate_hedge_signals(self,
                              prices: Dict[str, np.ndarray],
                              current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Hedge sinyalleri üretir.
        """
        if not self.enabled or len(self.pairs) < 2:
            return {"signals": [], "positions": {}}
        
        signals = []
        positions = {}
        
        # Co-integration analysis
        price1 = prices.get(self.pairs[0], [])
        price2 = prices.get(self.pairs[1], [])
        
        if len(price1) > self.lookback and len(price2) > self.lookback:
            coint_result = self.calculate_cointegration(price1[-self.lookback:], price2[-self.lookback:])
            
            if coint_result["cointegrated"]:
                # Spread calculation
                current_price1 = current_prices.get(self.pairs[0], 0)
                current_price2 = current_prices.get(self.pairs[1], 0)
                
                if current_price1 > 0 and current_price2 > 0:
                    spread = self.calculate_spread(
                        current_price1, current_price2,
                        coint_result["beta"], coint_result["intercept"]
                    )
                    
                    # Z-score
                    residuals = coint_result["residuals"]
                    z_score = (residuals[-1] - np.mean(residuals)) / np.std(residuals)
                    
                    # Signal generation
                    if abs(z_score) > self.z_threshold:
                        if z_score > 0:  # Spread is wide, short pair1, long pair2
                            signals.append({
                                "type": "spread_short",
                                "pair1": self.pairs[0],
                                "pair2": self.pairs[1],
                                "z_score": z_score,
                                "spread": spread
                            })
                        else:  # Spread is narrow, long pair1, short pair2
                            signals.append({
                                "type": "spread_long",
                                "pair1": self.pairs[0],
                                "pair2": self.pairs[1],
                                "z_score": z_score,
                                "spread": spread
                            })
        
        return {"signals": signals, "positions": positions}

    def apply(self, prices: Dict[str, np.ndarray], current_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Ana hedge overlay fonksiyonu.
        """
        if not self.enabled:
            return {"enabled": False}
        
        hedge_result = self.generate_hedge_signals(prices, current_prices)
        
        return {
            "enabled": True,
            "signals": hedge_result["signals"],
            "positions": hedge_result["positions"],
            "pairs": self.pairs
        } 
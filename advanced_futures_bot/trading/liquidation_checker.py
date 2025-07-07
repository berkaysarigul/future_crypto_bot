import logging
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger("LiquidationChecker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class LiquidationChecker:
    """
    Production-level margin usage & liquidation kontrolü.
    - Gerçek margin, liquidation, auto-flat
    - Exception handling, logging, edge-case yönetimi
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.liquidation_buffer = self.config.get("liquidation_buffer", 0.01)
        self.margin_threshold = self.config.get("margin_threshold", 0.8)
        self.auto_flat = self.config.get("auto_flat", True)
        self.min_margin = self.config.get("min_margin", 1.0)

    def calculate_margin_ratio(self,
                              account_balance: float,
                              position_value: float,
                              unrealized_pnl: float,
                              leverage: float) -> float:
        try:
            total_margin = position_value / max(leverage, 1e-4)
            available_balance = account_balance + unrealized_pnl
            margin_ratio = total_margin / max(available_balance, 1e-4)
            return margin_ratio
        except Exception as e:
            logger.error(f"Margin ratio hesaplama hatası: {e}")
            return 1.0

    def check_liquidation_risk(self,
                              account_balance: float,
                              position_value: float,
                              unrealized_pnl: float,
                              leverage: float,
                              current_price: float,
                              liquidation_price: float) -> Dict[str, Any]:
        try:
            margin_ratio = self.calculate_margin_ratio(account_balance, position_value, unrealized_pnl, leverage)
            price_to_liquidation = abs(current_price - liquidation_price) / max(current_price, 1e-4)
            buffer_breach = price_to_liquidation < self.liquidation_buffer
            margin_breach = margin_ratio > self.margin_threshold
            low_margin = (position_value / max(leverage, 1e-4)) < self.min_margin
            risk_level = "low"
            if buffer_breach or margin_breach or low_margin:
                risk_level = "high"
            elif margin_ratio > self.margin_threshold * 0.8:
                risk_level = "medium"
            should_flat = buffer_breach or (margin_breach and self.auto_flat) or low_margin
            if should_flat:
                logger.warning(f"Likidasyon riski! margin_ratio={margin_ratio:.4f}, buffer_breach={buffer_breach}, low_margin={low_margin}")
            return {
                "margin_ratio": margin_ratio,
                "buffer_breach": buffer_breach,
                "margin_breach": margin_breach,
                "low_margin": low_margin,
                "risk_level": risk_level,
                "should_flat": should_flat
            }
        except Exception as e:
            logger.error(f"Likidasyon risk kontrol hatası: {e}")
            return {"margin_ratio": 1.0, "should_flat": True, "risk_level": "error"}

    def check(self,
              account_balance: float,
              position_value: float,
              unrealized_pnl: float,
              leverage: float,
              current_price: float,
              liquidation_price: float) -> bool:
        try:
            risk_info = self.check_liquidation_risk(
                account_balance, position_value, unrealized_pnl, 
                leverage, current_price, liquidation_price
            )
            return risk_info["should_flat"]
        except Exception as e:
            logger.error(f"Likidasyon check hatası: {e}")
            return True 
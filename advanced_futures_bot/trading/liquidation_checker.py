from typing import Any, Dict, Optional
import numpy as np

class LiquidationChecker:
    """
    Margin usage ve liquidation kontrolü.
    - Liquidation buffer
    - Otomatik flat
    - Risk kontrolü
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.liquidation_buffer = self.config.get("liquidation_buffer", 0.01)
        self.margin_threshold = self.config.get("margin_threshold", 0.8)
        self.auto_flat = self.config.get("auto_flat", True)

    def calculate_margin_ratio(self,
                              account_balance: float,
                              position_value: float,
                              unrealized_pnl: float,
                              leverage: float) -> float:
        """
        Margin ratio hesaplama.
        """
        total_margin = position_value / leverage
        available_balance = account_balance + unrealized_pnl
        margin_ratio = total_margin / available_balance if available_balance > 0 else 1.0
        return margin_ratio

    def check_liquidation_risk(self,
                              account_balance: float,
                              position_value: float,
                              unrealized_pnl: float,
                              leverage: float,
                              current_price: float,
                              liquidation_price: float) -> Dict[str, Any]:
        """
        Liquidation risk kontrolü.
        """
        margin_ratio = self.calculate_margin_ratio(account_balance, position_value, unrealized_pnl, leverage)
        # Liquidation buffer kontrolü
        price_to_liquidation = abs(current_price - liquidation_price) / current_price
        buffer_breach = price_to_liquidation < self.liquidation_buffer
        # Margin threshold kontrolü
        margin_breach = margin_ratio > self.margin_threshold
        # Risk seviyesi
        risk_level = "low"
        if buffer_breach or margin_breach:
            risk_level = "high"
        elif margin_ratio > self.margin_threshold * 0.8:
            risk_level = "medium"
        return {
            "margin_ratio": margin_ratio,
            "buffer_breach": buffer_breach,
            "margin_breach": margin_breach,
            "risk_level": risk_level,
            "should_flat": buffer_breach or (margin_breach and self.auto_flat)
        }

    def check(self,
              account_balance: float,
              position_value: float,
              unrealized_pnl: float,
              leverage: float,
              current_price: float,
              liquidation_price: float) -> bool:
        """
        Ana liquidation check fonksiyonu.
        Returns: True if should flat position
        """
        risk_info = self.check_liquidation_risk(
            account_balance, position_value, unrealized_pnl, 
            leverage, current_price, liquidation_price
        )
        return risk_info["should_flat"] 
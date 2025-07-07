from typing import Any, Dict, Optional
import numpy as np

class PositionManager:
    """
    Position sizing ve dinamik kaldıraç ayarlama.
    - Risk yönetimi
    - Margin hesaplamaları
    - Dinamik kaldıraç
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_position = self.config.get("max_position", 0.5)
        self.base_leverage = self.config.get("leverage", 10)
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.current_position = 0.0
        self.current_leverage = self.base_leverage

    def calculate_position_size(self, 
                               account_balance: float,
                               current_price: float,
                               stop_loss_pct: float = 0.02) -> float:
        """
        Risk-based position sizing.
        """
        risk_amount = account_balance * self.risk_per_trade
        position_value = risk_amount / stop_loss_pct
        position_size = position_value / current_price
        # Max position limit
        max_size = account_balance * self.max_position / current_price
        return min(position_size, max_size)

    def adjust_leverage(self, 
                       volatility: float,
                       regime: str = "trend") -> float:
        """
        Dinamik kaldıraç ayarlama.
        """
        base = self.base_leverage
        if regime == "volatile":
            base *= 0.5  # Volatilitede kaldıracı azalt
        elif regime == "range":
            base *= 0.8  # Range'de orta kaldıraç
        # Volatiliteye göre ek ayarlama
        vol_factor = 1.0 / (1.0 + volatility)
        self.current_leverage = base * vol_factor
        return self.current_leverage

    def adjust(self,
               action: int,
               account_balance: float,
               current_price: float,
               volatility: float = 0.02,
               regime: str = "trend") -> Dict[str, Any]:
        """
        Ana position adjustment fonksiyonu.
        """
        # Kaldıraç ayarla
        leverage = self.adjust_leverage(volatility, regime)
        # Position size hesapla
        position_size = self.calculate_position_size(account_balance, current_price)
        # Action'a göre position
        if action == 1:  # Long
            self.current_position = position_size
        elif action == -1:  # Short
            self.current_position = -position_size
        else:  # Flat
            self.current_position = 0.0
        return {
            "position": self.current_position,
            "leverage": leverage,
            "position_value": abs(self.current_position) * current_price
        } 
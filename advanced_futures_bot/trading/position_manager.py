import logging
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger("PositionManager")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class PositionManager:
    """
    Production-level position sizing & leverage management.
    - Gerçek risk yönetimi, margin, kaldıraç
    - Edge-case, logging, exception handling
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_position = self.config.get("max_position", 0.5)
        self.base_leverage = self.config.get("leverage", 10)
        self.risk_per_trade = self.config.get("risk_per_trade", 0.02)
        self.min_balance = self.config.get("min_balance", 10.0)
        self.current_position = 0.0
        self.current_leverage = self.base_leverage

    def calculate_position_size(self, 
                               account_balance: float,
                               current_price: float,
                               stop_loss_pct: float = 0.02) -> float:
        try:
            if account_balance < self.min_balance:
                logger.warning(f"Yetersiz bakiye: {account_balance}")
                return 0.0
            risk_amount = account_balance * self.risk_per_trade
            position_value = risk_amount / max(stop_loss_pct, 1e-4)
            position_size = position_value / max(current_price, 1e-4)
            max_size = account_balance * self.max_position / max(current_price, 1e-4)
            size = min(position_size, max_size)
            if size <= 0:
                logger.warning(f"Pozisyon büyüklüğü sıfır veya negatif: {size}")
            return size
        except Exception as e:
            logger.error(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return 0.0

    def adjust_leverage(self, 
                       volatility: float,
                       regime: str = "trend") -> float:
        try:
            base = self.base_leverage
            if regime == "volatile":
                base *= 0.5
            elif regime == "range":
                base *= 0.8
            vol_factor = 1.0 / (1.0 + max(volatility, 1e-4))
            leverage = base * vol_factor
            leverage = np.clip(leverage, 1.0, 100.0)
            logger.info(f"Kaldıraç ayarlandı: {leverage:.2f} (regime={regime}, vol={volatility:.4f})")
            self.current_leverage = leverage
            return leverage
        except Exception as e:
            logger.error(f"Kaldıraç ayarlama hatası: {e}")
            return self.base_leverage

    def adjust(self,
               action: int,
               account_balance: float,
               current_price: float,
               volatility: float = 0.02,
               regime: str = "trend",
               stop_loss_pct: float = 0.02) -> Dict[str, Any]:
        try:
            leverage = self.adjust_leverage(volatility, regime)
            position_size = self.calculate_position_size(account_balance, current_price, stop_loss_pct)
            if action == 1:  # Long
                self.current_position = position_size
                logger.info(f"Long pozisyon açıldı: {self.current_position:.4f} BTC, Kaldıraç: {leverage:.2f}")
            elif action == -1:  # Short
                self.current_position = -position_size
                logger.info(f"Short pozisyon açıldı: {self.current_position:.4f} BTC, Kaldıraç: {leverage:.2f}")
            else:  # Flat
                logger.info(f"Pozisyon kapatıldı. Önceki pozisyon: {self.current_position:.4f} BTC")
                self.current_position = 0.0
            position_value = abs(self.current_position) * current_price
            margin = position_value / max(leverage, 1e-4)
            return {
                "position": self.current_position,
                "leverage": leverage,
                "position_value": position_value,
                "margin": margin
            }
        except Exception as e:
            logger.error(f"Pozisyon ayarlama hatası: {e}")
            return {"position": 0.0, "leverage": self.base_leverage, "position_value": 0.0, "margin": 0.0} 
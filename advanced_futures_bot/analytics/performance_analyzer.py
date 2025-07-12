import numpy as np
import logging
from typing import List, Dict, Any

class PerformanceAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger("PerformanceAnalyzer")

    def sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        try:
            excess_returns = np.array(returns) - risk_free_rate
            return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
        except Exception as e:
            self.logger.error(f"Sharpe ratio hesaplanamadı: {e}")
            return 0.0

    def sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        try:
            downside = np.std([r for r in returns if r < risk_free_rate]) + 1e-8
            excess_returns = np.mean(returns) - risk_free_rate
            return excess_returns / downside
        except Exception as e:
            self.logger.error(f"Sortino ratio hesaplanamadı: {e}")
            return 0.0

    def max_drawdown(self, equity_curve: List[float]) -> float:
        try:
            curve = np.array(equity_curve)
            drawdown = np.maximum.accumulate(curve) - curve
            return np.max(drawdown)
        except Exception as e:
            self.logger.error(f"Max drawdown hesaplanamadı: {e}")
            return 0.0

    def win_loss_ratio(self, trades: List[Dict[str, Any]]) -> float:
        try:
            wins = sum(1 for t in trades if t["pnl"] > 0)
            losses = sum(1 for t in trades if t["pnl"] <= 0)
            return wins / (losses + 1e-8)
        except Exception as e:
            self.logger.error(f"Win/Loss oranı hesaplanamadı: {e}")
            return 0.0 
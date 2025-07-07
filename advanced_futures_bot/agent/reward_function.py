from typing import Any, Dict, Optional
import numpy as np

class RewardFunction:
    """
    Futures RL için reward shaping:
    - PnL
    - Funding fee
    - Liquidation penalty
    - VaR penalty
    - Sentiment bonus
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.funding_fee_penalty = self.config.get("funding_fee_penalty", 0.5)
        self.liquidation_penalty = self.config.get("liquidation_buffer", 0.01)
        self.var_penalty = self.config.get("var_penalty", 0.2)
        self.sentiment_bonus = self.config.get("sentiment_bonus", 0.1)

    def compute(self,
                pnl: float,
                funding_fee: float,
                is_liquidated: bool,
                var: float,
                sentiment_score: float) -> float:
        reward = pnl
        reward -= abs(funding_fee) * self.funding_fee_penalty
        if is_liquidated:
            reward -= self.liquidation_penalty
        reward -= abs(var) * self.var_penalty
        reward += sentiment_score * self.sentiment_bonus
        return reward 
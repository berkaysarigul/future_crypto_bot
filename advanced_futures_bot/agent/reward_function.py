import logging
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger("RewardFunction")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class RewardFunction:
    """
    Production-level reward shaping:
    - PnL, funding fee, liquidation, VaR, sentiment shaping
    - Edge-case handling, logging
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.funding_fee_penalty = self.config.get("funding_fee_penalty", 0.5)
        self.liquidation_penalty = self.config.get("liquidation_buffer", 0.01)
        self.var_penalty = self.config.get("var_penalty", 0.2)
        self.sentiment_bonus = self.config.get("sentiment_bonus", 0.1)
        self.max_drawdown_penalty = self.config.get("max_drawdown_penalty", 0.3)
        self.reward_clip = self.config.get("reward_clip", 10.0)
        self.prev_balance = None
        self.max_drawdown = 0.0

    def compute(self,
                pnl: float,
                funding_fee: float,
                is_liquidated: bool,
                var: float,
                sentiment_score: float,
                account_balance: Optional[float] = None) -> float:
        reward = pnl
        reward -= abs(funding_fee) * self.funding_fee_penalty
        if is_liquidated:
            logger.warning("Liquidation penalty applied!")
            reward -= self.liquidation_penalty
        reward -= abs(var) * self.var_penalty
        reward += sentiment_score * self.sentiment_bonus
        # Max drawdown penalty
        if account_balance is not None:
            if self.prev_balance is not None:
                drawdown = max(0, self.prev_balance - account_balance)
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
                reward -= self.max_drawdown * self.max_drawdown_penalty
            self.prev_balance = account_balance
        # Reward clipping
        reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        logger.info(f"Reward computed: {reward:.4f} (PnL={pnl}, Funding={funding_fee}, Liquidated={is_liquidated}, VaR={var}, Sentiment={sentiment_score}, Drawdown={self.max_drawdown})")
        return reward 
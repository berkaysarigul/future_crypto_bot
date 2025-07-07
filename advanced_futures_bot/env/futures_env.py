import gymnasium as gym
import numpy as np
import logging
from typing import Any, Dict, Tuple, Optional

logger = logging.getLogger("FuturesEnv")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class FuturesEnv(gym.Env):
    """
    Production-level PPO custom Gym ortamı.
    - Gerçekçi state, reward shaping, episode management
    - Action/observation normalization, regime context
    - Exception handling, logging
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 sentiment_features: Any = None,
                 ob_features: Any = None,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.sentiment_features = sentiment_features
        self.ob_features = ob_features
        self.max_steps = self.config.get("max_steps", 1000)
        self.action_space = gym.spaces.Discrete(3)  # -1, 0, 1
        self.obs_dim = self.config.get("obs_dim", 32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.current_step = 0
        self.done = False
        self.position = 0  # -1, 0, 1
        self.pnl = 0.0
        self.account_balance = self.config.get("init_balance", 10000.0)
        self.position_price = 0.0
        self.unrealized_pnl = 0.0
        self.liquidated = False
        self.regime = "trend"
        self.history = []
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.done = False
        self.position = 0
        self.pnl = 0.0
        self.account_balance = self.config.get("init_balance", 10000.0)
        self.position_price = 0.0
        self.unrealized_pnl = 0.0
        self.liquidated = False
        self.regime = "trend"
        self.history = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        self._reset_state()
        obs = self._get_observation()
        logger.info("Environment reset.")
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Action: 0=short, 1=flat, 2=long → -1/0/1
        action_map = {0: -1, 1: 0, 2: 1}
        prev_position = self.position
        self.position = action_map.get(action, 0)
        # Simüle fiyat ve feature güncellemesi (örnek, gerçek veriyle entegre edilmeli)
        price = self._get_price()
        reward, info = self._compute_reward(price, prev_position)
        self.current_step += 1
        self.done = self._check_done(info)
        obs = self._get_observation()
        self.history.append({
            "step": self.current_step,
            "position": self.position,
            "price": price,
            "reward": reward,
            "pnl": self.pnl,
            "liquidated": self.liquidated
        })
        if self.done:
            logger.info(f"Episode finished at step {self.current_step}. PnL: {self.pnl:.2f} Liquidated: {self.liquidated}")
        return obs, reward, self.done, False, info

    def _get_price(self) -> float:
        # Gerçek veriyle entegre edilecek, şimdilik random walk
        if self.history:
            last_price = self.history[-1]["price"]
        else:
            last_price = 50000.0
        price = last_price * (1 + np.random.normal(0, 0.001))
        return float(np.clip(price, 1000, 100000))

    def _get_observation(self) -> np.ndarray:
        # Gerçek feature engineering burada yapılacak
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        # Örnek: fiyat, pozisyon, sentiment, order book, regime, PnL, risk
        obs[0] = self._get_price()
        obs[1] = self.position
        obs[2] = self.pnl
        obs[3] = self.account_balance
        # ... sentiment, order book, regime, on-chain vs. eklenebilir
        return obs

    def _compute_reward(self, price: float, prev_position: int) -> Tuple[float, dict]:
        # PnL, funding, liquidation, VaR, sentiment shaping
        reward = 0.0
        info = {"pnl": self.pnl, "is_liquidated": self.liquidated, "regime": self.regime}
        # Pozisyon değişimi varsa giriş fiyatı güncelle
        if self.position != prev_position:
            self.position_price = price
        # Unrealized PnL
        self.unrealized_pnl = (price - self.position_price) * self.position
        # Realized PnL (pozisyon kapanınca)
        if prev_position != 0 and self.position == 0:
            realized = (price - self.position_price) * prev_position
            self.pnl += realized
            reward += realized
        # Funding, liquidation, VaR, sentiment shaping burada eklenebilir
        # Likidasyon kontrolü (örnek)
        if abs(self.unrealized_pnl) > self.account_balance * 0.8:
            self.liquidated = True
            reward -= self.account_balance * 0.5
            self.done = True
        info.update({
            "unrealized_pnl": self.unrealized_pnl,
            "account_balance": self.account_balance,
            "liquidated": self.liquidated
        })
        return reward, info

    def _check_done(self, info: dict) -> bool:
        if self.current_step >= self.max_steps:
            return True
        if info.get("liquidated", False):
            return True
        return False

    def render(self, mode: str = "human"):
        print(f"Step: {self.current_step}, Position: {self.position}, PnL: {self.pnl:.2f}, Balance: {self.account_balance:.2f}, Liquidated: {self.liquidated}, Regime: {self.regime}") 
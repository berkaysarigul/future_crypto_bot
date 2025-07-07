import unittest
import numpy as np

from data.data_fetcher import DataFetcher
from data.sentiment_deepseek import DeepSeekSentiment
from data.order_book_handler import OrderBookHandler
from env.futures_env import FuturesEnv
from agent.hybrid_policy import HybridPolicy
from agent.reward_function import RewardFunction
from agent.hyperopt import HyperOpt
from agent.regime_switcher import RegimeSwitcher
from agent.continual_trainer import ContinualTrainer
from trading.position_manager import PositionManager
from trading.liquidation_checker import LiquidationChecker
from trading.execution_engine import ExecutionEngine
from trading.hedge_overlay import HedgeOverlay
from utils.logger import Logger
from utils.explainable_ai import ExplainableAI

class TestFuturesBot(unittest.TestCase):
    def setUp(self):
        self.dummy_config = {"binance": {"symbol": "BTCUSDT"}, "data": {}}

    def test_data_fetcher(self):
        fetcher = DataFetcher(self.dummy_config)
        self.assertTrue(hasattr(fetcher, "fetch_ohlcv"))

    def test_sentiment(self):
        sentiment = DeepSeekSentiment(api_key="test")
        result = sentiment.analyze(["BTC yükseliyor!"])
        self.assertIsInstance(result, list)

    def test_order_book(self):
        ob = OrderBookHandler({"bids": [["100", "1"]], "asks": [["101", "1"]]})
        self.assertIsInstance(ob.depth_imbalance(), float)

    def test_env(self):
        env = FuturesEnv()
        obs, _ = env.reset()
        self.assertEqual(obs.shape[0], 32)

    def test_hybrid_policy(self):
        model = HybridPolicy()
        x = np.zeros((1, 10, 32), dtype=np.float32)
        regime = np.zeros((1, 10, 4), dtype=np.float32)
        import torch
        logits, value, hidden = model(torch.tensor(x), torch.tensor(regime))
        self.assertEqual(logits.shape[-1], 3)

    def test_reward_function(self):
        reward_fn = RewardFunction()
        reward = reward_fn.compute(1.0, 0.01, False, 0.02, 0.5)
        self.assertIsInstance(reward, float)

    def test_hyperopt(self):
        class DummyTrainer:
            def train_with_params(self, params):
                return {"reward": 1.0}
        hyperopt = HyperOpt(DummyTrainer(), method="optuna")
        self.assertTrue(callable(hyperopt.optimize))

    def test_regime_switcher(self):
        regime = RegimeSwitcher()
        price_series = np.linspace(100, 110, 50)
        detected = regime.detect(price_series)
        self.assertIn(detected, ["trend", "range", "volatile"])

    def test_continual_trainer(self):
        class DummyEnv:
            def reset(self): return (np.zeros(32), {})
            def step(self, action): return (np.zeros(32), 0.0, True, False, {"pnl": 0.0})
        class DummyPolicy:
            def act(self, state, regime, hidden=None): return 0, 0.0, None
        class DummyReward:
            def compute(self, **kwargs): return 1.0
        class DummyRegime:
            def get_context(self): return np.zeros(4)
        trainer = ContinualTrainer(DummyEnv(), DummyPolicy(), DummyReward(), DummyRegime())
        self.assertTrue(callable(trainer.train))

    def test_position_manager(self):
        pm = PositionManager()
        result = pm.adjust(1, 1000, 100)
        self.assertIn("position", result)

    def test_liquidation_checker(self):
        lc = LiquidationChecker()
        should_flat = lc.check(1000, 500, 0, 10, 100, 90)
        self.assertIsInstance(should_flat, bool)

    def test_execution_engine(self):
        ee = ExecutionEngine()
        ob = {"bids": [["100", "1"]], "asks": [["101", "1"]]}
        result = ee.run(1, "buy", ob)
        self.assertIn("filled_size", result)

    def test_hedge_overlay(self):
        ho = HedgeOverlay({"hedge_enabled": True})
        prices = {"BTCUSDT": np.linspace(100, 110, 120), "ETHUSDT": np.linspace(50, 60, 120)}
        current_prices = {"BTCUSDT": 110, "ETHUSDT": 60}
        result = ho.apply(prices, current_prices)
        self.assertIn("enabled", result)

    def test_explainable_ai(self):
        class DummyModel:
            def __call__(self, x, regime=None, hidden=None):
                return torch.zeros((x.shape[0], 3)), torch.zeros(x.shape[0]), None
        x = np.zeros((5, 32), dtype=np.float32)
        eai = ExplainableAI()
        result = eai.explain(DummyModel(), x, x)
        self.assertIn("shap", result)

if __name__ == "__main__":
    unittest.main() 
import unittest
import numpy as np
import pandas as pd
import torch
import logging
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_fetcher import DataFetcher
from data.news_manager import NewsManager
from sentiment.sentiment_deepseek import SentimentAnalyzer
from trading.order_book_handler import OrderBookHandler
from trading.futures_env import FuturesEnvironment
from models.hybrid_policy import HybridPolicy
from models.reward_function import RewardFunction
from optimization.hyperopt import HyperoptOptimizer
from trading.regime_switcher import RegimeSwitcher
from training.continual_trainer import ContinualTrainer
from risk.position_manager import PositionManager
from risk.liquidation_checker import LiquidationChecker
from trading.execution_engine import ExecutionEngine
from trading.hedge_overlay import HedgeOverlay
from utils.logger import TradingLogger
from utils.explainable_ai import ExplainableAI
from backtest.backtester import Backtester
from env.futures_env import FuturesEnv
from models.model_manager import ModelManager
from analytics.performance_analyzer import PerformanceAnalyzer
from data.data_validator import DataValidator
from config.config_validator import ConfigValidator

# Setup logging for tests
logging.basicConfig(level=logging.ERROR)

class TestDataFetcher(unittest.TestCase):
    """Test DataFetcher module."""
    
    def setUp(self):
        self.config = {
            "binance": {
                "api_key": "test_key",
                "api_secret": "test_secret",
                "testnet": True
            },
            "symbols": ["BTCUSDT"],
            "timeframes": ["1m", "5m", "15m"]
        }
        self.data_fetcher = DataFetcher(self.config)
    
    def test_initialization(self):
        """Test DataFetcher initialization."""
        self.assertIsNotNone(self.data_fetcher)
        self.assertEqual(self.data_fetcher.symbols, ["BTCUSDT"])
    
    @patch('data.data_fetcher.Client')
    def test_fetch_klines(self, mock_client):
        """Test klines fetching."""
        # Mock response
        mock_response = [
            [1625097600000, "35000.00", "35100.00", "34900.00", "35050.00", "100.5", 1625097899999, "3510000.00", 1000, "50.5", "49.5", "0"],
            [1625097900000, "35050.00", "35200.00", "35000.00", "35150.00", "150.2", 1625098199999, "5270000.00", 1500, "75.1", "75.1", "0"]
        ]
        mock_client.return_value.get_klines.return_value = mock_response
        
        result = self.data_fetcher.fetch_klines("BTCUSDT", "1m", limit=2)
        
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertIn("open", result.columns)
        self.assertIn("high", result.columns)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create mock data
        mock_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [103, 104, 105],
            'low': [99, 100, 101],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        
        processed = self.data_fetcher.preprocess_data(mock_data)
        
        self.assertIsNotNone(processed)
        self.assertIn("returns", processed.columns)
        self.assertIn("volatility", processed.columns)

class TestSentimentAnalyzer(unittest.TestCase):
    """Test SentimentAnalyzer module."""
    
    def setUp(self):
        self.config = {
            "deepseek": {
                "api_key": "test_key",
                "model": "deepseek-chat",
                "max_tokens": 100
            }
        }
        self.analyzer = SentimentAnalyzer(self.config)
    
    @patch('sentiment.sentiment_deepseek.requests.post')
    def test_analyze_sentiment(self, mock_post):
        """Test sentiment analysis."""
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "positive"}}]
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        text = "Bitcoin is showing strong bullish momentum"
        result = self.analyzer.analyze_sentiment(text)
        
        self.assertIsNotNone(result)
        self.assertIn("sentiment", result)
    
    def test_batch_analyze(self):
        """Test batch sentiment analysis."""
        texts = [
            "Bitcoin is bullish",
            "Market is bearish",
            "Neutral market conditions"
        ]
        
        with patch.object(self.analyzer, 'analyze_sentiment') as mock_analyze:
            mock_analyze.return_value = {"sentiment": "neutral", "confidence": 0.5}
            results = self.analyzer.batch_analyze(texts)
            
            self.assertEqual(len(results), 3)
            self.assertIn("sentiment", results[0])

class TestOrderBookHandler(unittest.TestCase):
    """Test OrderBookHandler module."""
    
    def setUp(self):
        self.config = {
            "order_book": {
                "depth": 20,
                "update_interval": 1.0
            }
        }
        self.handler = OrderBookHandler(self.config)
    
    def test_process_order_book(self):
        """Test order book processing."""
        # Mock order book data
        mock_order_book = {
            "bids": [[35000, 1.5], [34999, 2.0], [34998, 1.0]],
            "asks": [[35001, 1.0], [35002, 2.5], [35003, 1.5]]
        }
        
        result = self.handler.process_order_book(mock_order_book)
        
        self.assertIsNotNone(result)
        self.assertIn("bid_volume", result)
        self.assertIn("ask_volume", result)
        self.assertIn("spread", result)
    
    def test_calculate_depth(self):
        """Test depth calculation."""
        mock_order_book = {
            "bids": [[35000, 1.5], [34999, 2.0], [34998, 1.0]],
            "asks": [[35001, 1.0], [35002, 2.5], [35003, 1.5]]
        }
        
        depth = self.handler.calculate_depth(mock_order_book, 1.0)
        
        self.assertIsNotNone(depth)
        self.assertGreater(depth, 0)

class TestFuturesEnvironment(unittest.TestCase):
    """Test FuturesEnvironment module."""
    
    def setUp(self):
        self.config = {
            "environment": {
                "initial_balance": 10000,
                "max_position_size": 0.1,
                "transaction_fee": 0.001
            }
        }
        self.env = FuturesEnvironment(self.config)
    
    def test_reset(self):
        """Test environment reset."""
        state = self.env.reset()
        
        self.assertIsNotNone(state)
        self.assertEqual(self.env.balance, 10000)
        self.assertEqual(self.env.position, 0)
    
    def test_step(self):
        """Test environment step."""
        self.env.reset()
        
        # Mock market data
        mock_data = {
            "price": 35000,
            "volume": 1000,
            "order_book": {"bids": [[34999, 1]], "asks": [[35001, 1]]}
        }
        
        action = 1  # Buy
        next_state, reward, done, info = self.env.step(action, mock_data)
        
        self.assertIsNotNone(next_state)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

class TestHybridPolicy(unittest.TestCase):
    """Test HybridPolicy module."""
    
    def setUp(self):
        self.config = {
            "policy": {
                "state_dim": 10,
                "action_dim": 3,
                "hidden_dim": 64,
                "lstm_hidden_dim": 32,
                "transformer_dim": 128
            }
        }
        self.policy = HybridPolicy(self.config)
    
    def test_forward(self):
        """Test policy forward pass."""
        batch_size = 4
        seq_len = 10
        state_dim = 10
        
        state = torch.randn(batch_size, seq_len, state_dim)
        
        with torch.no_grad():
            logits, value, hidden = self.policy(state)
        
        self.assertEqual(logits.shape, (batch_size, 3))
        self.assertEqual(value.shape, (batch_size, 1))
        self.assertIsNotNone(hidden)
    
    def test_get_action(self):
        """Test action selection."""
        state = torch.randn(1, 10, 10)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state)
        
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)
        self.assertIsInstance(value, float)

class TestRewardFunction(unittest.TestCase):
    """Test RewardFunction module."""
    
    def setUp(self):
        self.config = {
            "reward": {
                "pnl_weight": 1.0,
                "drawdown_penalty": 0.5,
                "sentiment_weight": 0.3
            }
        }
        self.reward_fn = RewardFunction(self.config)
    
    def test_calculate_reward(self):
        """Test reward calculation."""
        prev_state = {
            "balance": 10000,
            "position": 0,
            "price": 35000
        }
        
        current_state = {
            "balance": 10100,
            "position": 0.1,
            "price": 35100
        }
        
        action = 1
        sentiment = {"sentiment": "positive", "confidence": 0.8}
        
        reward = self.reward_fn.calculate_reward(prev_state, current_state, action, sentiment)
        
        self.assertIsInstance(reward, float)
        self.assertGreater(reward, -1000)  # Reasonable bounds

class TestHyperoptOptimizer(unittest.TestCase):
    """Test HyperoptOptimizer module."""
    
    def setUp(self):
        self.config = {
            "hyperopt": {
                "n_trials": 5,
                "timeout": 60
            }
        }
        self.optimizer = HyperoptOptimizer(self.config)
    
    def test_define_search_space(self):
        """Test search space definition."""
        space = self.optimizer.define_search_space()
        
        self.assertIsNotNone(space)
        self.assertIn("learning_rate", space)
        self.assertIn("hidden_dim", space)
    
    @patch('optimization.hyperopt.optuna.create_study')
    def test_optimize(self, mock_create_study):
        """Test optimization process."""
        mock_study = Mock()
        mock_study.optimize.return_value = None
        mock_create_study.return_value = mock_study
        
        def objective(trial):
            return trial.suggest_float("x", 0, 1)
        
        best_params = self.optimizer.optimize(objective)
        
        self.assertIsNotNone(best_params)

class TestRegimeSwitcher(unittest.TestCase):
    """Test RegimeSwitcher module."""
    
    def setUp(self):
        self.config = {
            "regime": {
                "lookback_period": 20,
                "volatility_threshold": 0.02
            }
        }
        self.switcher = RegimeSwitcher(self.config)
    
    def test_detect_regime(self):
        """Test regime detection."""
        # Create mock price data
        prices = np.array([100 + i + np.random.normal(0, 1) for i in range(50)])
        
        regime = self.switcher.detect_regime(prices)
        
        self.assertIn(regime, ["trend", "range", "volatile"])
    
    def test_get_regime_params(self):
        """Test regime parameter retrieval."""
        regime = "trend"
        params = self.switcher.get_regime_params(regime)
        
        self.assertIsNotNone(params)
        self.assertIn("learning_rate", params)

class TestContinualTrainer(unittest.TestCase):
    """Test ContinualTrainer module."""
    
    def setUp(self):
        self.config = {
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "epochs": 10
            }
        }
        self.trainer = ContinualTrainer(self.config)
    
    def test_prepare_batch(self):
        """Test batch preparation."""
        # Mock rollout data
        rollout_data = {
            "states": torch.randn(100, 10),
            "actions": torch.randint(0, 3, (100,)),
            "rewards": torch.randn(100),
            "values": torch.randn(100, 1)
        }
        
        batch = self.trainer.prepare_batch(rollout_data, batch_size=32)
        
        self.assertIsNotNone(batch)
        self.assertIn("states", batch)
        self.assertIn("actions", batch)
    
    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = torch.tensor([1.0, 0.5, -0.5, 1.0])
        values = torch.tensor([[0.5], [0.6], [0.4], [0.8]])
        
        gae = self.trainer.compute_gae(rewards, values, gamma=0.99, lam=0.95)
        
        self.assertEqual(gae.shape, rewards.shape)

class TestPositionManager(unittest.TestCase):
    """Test PositionManager module."""
    
    def setUp(self):
        self.config = {
            "position": {
                "max_position_size": 0.1,
                "max_leverage": 10,
                "stop_loss_pct": 0.05
            }
        }
        self.manager = PositionManager(self.config)
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        balance = 10000
        price = 35000
        confidence = 0.8
        
        size = self.manager.calculate_position_size(balance, price, confidence)
        
        self.assertGreater(size, 0)
        self.assertLessEqual(size, 0.1)  # Max position size
    
    def test_check_risk_limits(self):
        """Test risk limit checking."""
        position = 0.05
        balance = 10000
        price = 35000
        
        is_safe = self.manager.check_risk_limits(position, balance, price)
        
        self.assertIsInstance(is_safe, bool)

class TestLiquidationChecker(unittest.TestCase):
    """Test LiquidationChecker module."""
    
    def setUp(self):
        self.config = {
            "liquidation": {
                "margin_threshold": 0.8,
                "auto_flat_threshold": 0.9
            }
        }
        self.checker = LiquidationChecker(self.config)
    
    def test_check_liquidation_risk(self):
        """Test liquidation risk checking."""
        position = 0.1
        balance = 10000
        price = 35000
        entry_price = 34000
        
        risk = self.checker.check_liquidation_risk(position, balance, price, entry_price)
        
        self.assertIsInstance(risk, dict)
        self.assertIn("margin_ratio", risk)
        self.assertIn("liquidation_risk", risk)
    
    def test_should_auto_flat(self):
        """Test auto-flat decision."""
        margin_ratio = 0.95
        
        should_flat = self.checker.should_auto_flat(margin_ratio)
        
        self.assertIsInstance(should_flat, bool)

class TestExecutionEngine(unittest.TestCase):
    """Test ExecutionEngine module."""
    
    def setUp(self):
        self.config = {
            "execution": {
                "slippage_model": "linear",
                "latency_ms": 50,
                "partial_fill_prob": 0.1
            }
        }
        self.engine = ExecutionEngine(self.config)
    
    def test_simulate_execution(self):
        """Test execution simulation."""
        order = {
            "side": "buy",
            "size": 1.0,
            "price": 35000
        }
        
        order_book = {
            "bids": [[34999, 1], [34998, 2]],
            "asks": [[35001, 1], [35002, 2]]
        }
        
        result = self.engine.simulate_execution(order, order_book)
        
        self.assertIsNotNone(result)
        self.assertIn("executed_price", result)
        self.assertIn("executed_size", result)
        self.assertIn("slippage", result)
    
    def test_calculate_slippage(self):
        """Test slippage calculation."""
        order_size = 1.0
        order_book = {
            "bids": [[34999, 1], [34998, 2]],
            "asks": [[35001, 1], [35002, 2]]
        }
        
        slippage = self.engine.calculate_slippage("buy", order_size, order_book)
        
        self.assertIsInstance(slippage, float)
        self.assertGreaterEqual(slippage, 0)

class TestHedgeOverlay(unittest.TestCase):
    """Test HedgeOverlay module."""
    
    def setUp(self):
        self.config = {
            "hedge": {
                "enabled": True,
                "pairs": ["BTCUSDT", "ETHUSDT"],
                "lookback": 50
            }
        }
        self.overlay = HedgeOverlay(self.config)
    
    def test_calculate_cointegration(self):
        """Test cointegration calculation."""
        # Create correlated price series
        np.random.seed(42)
        price1 = np.cumsum(np.random.normal(0, 1, 100)) + 100
        price2 = price1 * 0.5 + np.random.normal(0, 0.1, 100)
        
        result = self.overlay.calculate_cointegration(price1, price2)
        
        self.assertIsNotNone(result)
        self.assertIn("cointegrated", result)
        self.assertIn("beta", result)
    
    def test_calculate_spread(self):
        """Test spread calculation."""
        price1 = 35000
        price2 = 17500
        beta = 0.5
        intercept = 0
        
        spread = self.overlay.calculate_spread(price1, price2, beta, intercept)
        
        self.assertIsInstance(spread, float)

class TestTradingLogger(unittest.TestCase):
    """Test TradingLogger module."""
    
    def setUp(self):
        self.config = {
            "log_dir": "test_logs",
            "use_wandb": False,
            "use_tensorboard": False
        }
        self.logger = TradingLogger(self.config)
    
    def test_log_episode_metrics(self):
        """Test episode metrics logging."""
        metrics = {
            "reward": 100.5,
            "pnl": 50.0,
            "episode_length": 100
        }
        
        self.logger.log_episode_metrics(1, metrics)
        
        self.assertEqual(len(self.logger.episode_metrics), 1)
        self.assertEqual(self.logger.episode_metrics[0]["episode"], 1)
    
    def test_log_trading_metrics(self):
        """Test trading metrics logging."""
        metrics = {
            "position": 0.1,
            "leverage": 2.0,
            "margin_ratio": 0.5
        }
        
        self.logger.log_trading_metrics(metrics)
        
        self.assertEqual(len(self.logger.trading_metrics), 1)
    
    def tearDown(self):
        """Clean up after tests."""
        self.logger.close()

class TestExplainableAI(unittest.TestCase):
    """Test ExplainableAI module."""
    
    def setUp(self):
        self.config = {
            "explainability_dir": "test_explainability",
            "feature_names": ["price", "volume", "rsi", "macd"],
            "max_explanation_samples": 10
        }
        self.explainer = ExplainableAI(self.config)
    
    def test_setup_lime_explainer(self):
        """Test LIME explainer setup."""
        training_data = np.random.randn(100, 4)
        
        success = self.explainer.setup_lime_explainer(training_data)
        
        # Will be False if LIME not available, but should not raise error
        self.assertIsInstance(success, bool)
    
    def test_analyze_feature_importance(self):
        """Test feature importance analysis."""
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.4, 0.3]]))
        
        X = np.random.randn(50, 4)
        y = np.random.randint(0, 3, 50)
        
        result = self.explainer.analyze_feature_importance(mock_model, X, y)
        
        self.assertIsNotNone(result)
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        if os.path.exists("test_explainability"):
            shutil.rmtree("test_explainability")

class TestBacktester(unittest.TestCase):
    def setUp(self):
        from backtest.backtester import Backtester
        from env.futures_env import FuturesEnv
        from trading.execution_engine import ExecutionEngine
        self.env = MagicMock(spec=FuturesEnv)
        self.executor = MagicMock(spec=ExecutionEngine)
        self.config = {"walk_forward": True}
        self.backtester = Backtester(self.env, self.executor, self.config)

    def test_run_walk_forward(self):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.backtester.run_walk_forward())
        self.assertIsNone(result)

class TestModelManager(unittest.TestCase):
    def setUp(self):
        from models.model_manager import ModelManager
        self.model_manager = ModelManager("/tmp/models")

    def test_save_and_load_checkpoint(self):
        self.model_manager.save_checkpoint({}, "test_model")
        model = self.model_manager.load_checkpoint("test_model")
        self.assertIsNone(model)

    def test_list_versions(self):
        versions = self.model_manager.list_versions()
        self.assertIsInstance(versions, list)

    def test_ab_test(self):
        result = self.model_manager.ab_test({}, {})
        self.assertIn(result, ["A", "B", "error"])

class TestPerformanceAnalyzer(unittest.TestCase):
    def setUp(self):
        from analytics.performance_analyzer import PerformanceAnalyzer
        self.analyzer = PerformanceAnalyzer()

    def test_sharpe_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.03]
        ratio = self.analyzer.sharpe_ratio(returns)
        self.assertIsInstance(ratio, float)

    def test_sortino_ratio(self):
        returns = [0.01, 0.02, -0.01, 0.03]
        ratio = self.analyzer.sortino_ratio(returns)
        self.assertIsInstance(ratio, float)

    def test_max_drawdown(self):
        curve = [100, 110, 105, 120, 90, 130]
        dd = self.analyzer.max_drawdown(curve)
        self.assertIsInstance(dd, float)

    def test_win_loss_ratio(self):
        trades = [{"pnl": 1}, {"pnl": -1}, {"pnl": 2}]
        wl = self.analyzer.win_loss_ratio(trades)
        self.assertIsInstance(wl, float)

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        from data.data_validator import DataValidator
        self.validator = DataValidator()
        self.data = [
            {"price": 100, "volume": 1},
            {"price": 101, "volume": 2},
            {"price": 100, "volume": 1},
        ]

    def test_check_sanity(self):
        self.assertTrue(self.validator.check_sanity(self.data))

    def test_detect_missing(self):
        missing = self.validator.detect_missing(self.data)
        self.assertIsInstance(missing, list)

    def test_clean_duplicates(self):
        cleaned = self.validator.clean_duplicates(self.data)
        self.assertIsInstance(cleaned, list)

class TestConfigValidator(unittest.TestCase):
    def setUp(self):
        from config.config_validator import ConfigValidator
        self.schema = {"param1": str, "param2": int}
        self.validator = ConfigValidator(self.schema)
        self.config_path = "advanced_futures_bot/config.yaml"

    def test_validate(self):
        valid = self.validator.validate(self.config_path)
        self.assertIsInstance(valid, bool)

class TestStrategyEvaluator(unittest.TestCase):
    def setUp(self):
        from strategies.strategy_evaluator import StrategyEvaluator
        self.evaluator = StrategyEvaluator([lambda x: x])
        self.data = [{"price": 100}, {"price": 101}]

    def test_evaluate(self):
        result = self.evaluator.evaluate(self.data)
        self.assertIsInstance(result, dict)

class TestStressTesting(unittest.TestCase):
    def setUp(self):
        from risk_management.stress_testing import StressTesting
        self.stress = StressTesting()
        self.data = [{"price": 100}, {"price": 90}]

    def test_flash_crash(self):
        result = self.stress.flash_crash(self.data)
        self.assertIsInstance(result, list)

    def test_black_swan(self):
        result = self.stress.black_swan(self.data)
        self.assertIsInstance(result, list)

class TestDriftDetector(unittest.TestCase):
    def setUp(self):
        from models.drift_detector import DriftDetector
        self.detector = DriftDetector(0.5)

    def test_detect(self):
        predictions = [0.1, 0.2, 0.3]
        actuals = [0.1, 0.2, 0.3]
        drift = self.detector.detect(predictions, actuals)
        self.assertIsInstance(drift, bool)

class TestFailoverHandler(unittest.TestCase):
    def setUp(self):
        from infrastructure.failover_handler import FailoverHandler
        self.alert_system = type('MockAlert', (), {"send_alert": lambda self, msg: None})()
        self.health_checker = object()
        self.handler = FailoverHandler(self.alert_system, self.health_checker)

    def test_handle_failover(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.handler.handle_failover("test error"))
        self.assertIsNone(result)

class TestNewsScraper(unittest.TestCase):
    def setUp(self):
        from external_data.news_scraper import NewsScraper
        self.scraper = NewsScraper(["https://example.com/api/news"])

    def test_fetch_news(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.scraper.fetch_news())
        self.assertIsInstance(result, list)

class TestSocialMediaMonitor(unittest.TestCase):
    def setUp(self):
        from external_data.social_media_monitor import SocialMediaMonitor
        self.monitor = SocialMediaMonitor({}, {})

    def test_fetch_twitter(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.monitor.fetch_twitter("btc"))
        self.assertIsInstance(result, list)

    def test_fetch_reddit(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.monitor.fetch_reddit("bitcoin"))
        self.assertIsInstance(result, list)

class TestWhaleTracker(unittest.TestCase):
    def setUp(self):
        from external_data.whale_tracker import WhaleTracker
        self.tracker = WhaleTracker("test_api_key")

    def test_fetch_whale_alerts(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.tracker.fetch_whale_alerts())
        self.assertIsInstance(result, list)

class TestFearGreedIndex(unittest.TestCase):
    def setUp(self):
        from external_data.fear_greed_index import FearGreedIndex
        self.index = FearGreedIndex("https://example.com/api/fear_greed")

    def test_fetch_index(self):
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self.index.fetch_index())
        self.assertIsInstance(result, dict)

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        from strategies.portfolio_optimizer import PortfolioOptimizer
        self.optimizer = PortfolioOptimizer()

    def test_optimize(self):
        assets = ["BTC", "ETH"]
        returns = {"BTC": [0.01, 0.02], "ETH": [0.03, 0.01]}
        weights = self.optimizer.optimize(assets, returns)
        self.assertIsInstance(weights, dict)

class TestTradeLogger(unittest.TestCase):
    def setUp(self):
        from utils.trade_logger import TradeLogger
        self.logger = TradeLogger(db_manager=None)

    def test_log_trade(self):
        trade = {"user": "test", "signal": "long", "pnl": 10, "funding": 0.1}
        self.logger.log_trade(trade)

class TestNewsManager(unittest.TestCase):
    """Test NewsManager module."""
    
    def setUp(self):
        self.config = {
            "news_sources": ["https://api.coingecko.com/api/v3/news"],
            "sentiment": {
                "api_url": "https://api.deepseek.com/v1/sentiment",
                "batch_size": 8
            },
            "update_interval": 3600
        }
        self.news_manager = NewsManager(self.config)
    
    def test_initialization(self):
        """Test NewsManager initialization."""
        self.assertIsNotNone(self.news_manager)
        self.assertEqual(self.news_manager.news_sources, ["https://api.coingecko.com/api/v3/news"])
        self.assertEqual(self.news_manager.update_interval, 3600)
    
    def test_get_empty_features(self):
        """Test empty features generation."""
        features = self.news_manager._get_empty_features()
        
        expected_keys = ['avg_sentiment', 'sentiment_std', 'positive_ratio', 
                        'negative_ratio', 'neutral_ratio', 'news_count', 'sentiment_momentum']
        
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], float)
    
    def test_create_sample_news(self):
        """Test sample news creation."""
        sample_news = self.news_manager._create_sample_news()
        
        self.assertIsInstance(sample_news, list)
        self.assertGreater(len(sample_news), 0)
        
        for news in sample_news:
            self.assertIn('title', news)
            self.assertIn('content', news)
            self.assertIn('source', news)
            self.assertIn('timestamp', news)
    
    @patch('pandas.read_csv')
    def test_get_latest_sentiment_features_with_data(self, mock_read_csv):
        """Test sentiment features extraction with mock data."""
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', '2024-01-01 11:00:00'],
            'sentiment': [0.5, -0.3]
        })
        mock_read_csv.return_value = mock_df
        
        features = self.news_manager.get_latest_sentiment_features(hours=24)
        
        self.assertIsInstance(features, dict)
        self.assertIn('avg_sentiment', features)
        self.assertIn('news_count', features)
    
    @patch('pandas.read_csv')
    def test_get_latest_sentiment_features_empty_data(self, mock_read_csv):
        """Test sentiment features extraction with empty data."""
        # Mock empty DataFrame
        mock_df = pd.DataFrame()
        mock_read_csv.return_value = mock_df
        
        features = self.news_manager.get_latest_sentiment_features(hours=24)
        
        self.assertIsInstance(features, dict)
        self.assertEqual(features['news_count'], 0)
        self.assertEqual(features['avg_sentiment'], 0.0)
    
    @patch('pathlib.Path.exists')
    def test_get_news_summary_file_not_exists(self, mock_exists):
        """Test news summary when file doesn't exist."""
        mock_exists.return_value = False
        
        summary = self.news_manager.get_news_summary()
        
        self.assertIn('error', summary)
        self.assertEqual(summary['error'], 'CSV dosyası bulunamadı')
    
    @patch('pandas.read_csv')
    @patch('pathlib.Path.exists')
    def test_get_news_summary_with_data(self, mock_exists, mock_read_csv):
        """Test news summary with mock data."""
        mock_exists.return_value = True
        
        # Mock DataFrame
        mock_df = pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00'],
            'source': ['test_source'],
            'sentiment': [0.5]
        })
        mock_read_csv.return_value = mock_df
        
        summary = self.news_manager.get_news_summary()
        
        self.assertIn('total_news', summary)
        self.assertIn('sentiment_stats', summary)
        self.assertEqual(summary['total_news'], 1)
    
    @patch('external_data.news_scraper.NewsScraper.fetch_news')
    @patch('data.sentiment_deepseek.DeepSeekSentiment.analyze')
    async def test_create_news_csv(self, mock_analyze, mock_fetch_news):
        """Test news CSV creation."""
        # Mock news data
        mock_fetch_news.return_value = [
            {
                'title': 'Test News',
                'content': 'Test Content',
                'source': 'test',
                'timestamp': '2024-01-01 10:00:00'
            }
        ]
        
        # Mock sentiment analysis
        mock_analyze.return_value = [{'sentiment': 0.5}]
        
        df = await self.news_manager._create_news_csv()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    @patch('external_data.news_scraper.NewsScraper.fetch_news')
    async def test_create_news_csv_no_news(self, mock_fetch_news):
        """Test news CSV creation when no news is fetched."""
        mock_fetch_news.return_value = []
        
        df = await self.news_manager._create_news_csv()
        
        self.assertIsInstance(df, pd.DataFrame)
        # Should create sample news when no real news is available
        self.assertGreater(len(df), 0)

def run_smoke_tests():
    """Run smoke tests for all modules."""
    print("Running smoke tests...")
    
    # Test basic functionality without external dependencies
    test_cases = [
        TestDataFetcher,
        TestSentimentAnalyzer,
        TestOrderBookHandler,
        TestFuturesEnvironment,
        TestHybridPolicy,
        TestRewardFunction,
        TestHyperoptOptimizer,
        TestRegimeSwitcher,
        TestContinualTrainer,
        TestPositionManager,
        TestLiquidationChecker,
        TestExecutionEngine,
        TestHedgeOverlay,
        TestTradingLogger,
        TestExplainableAI,
        TestBacktester,
        TestModelManager,
        TestPerformanceAnalyzer,
        TestDataValidator,
        TestConfigValidator,
        TestStrategyEvaluator,
        TestStressTesting,
        TestDriftDetector,
        TestFailoverHandler,
        TestNewsScraper,
        TestSocialMediaMonitor,
        TestWhaleTracker,
        TestFearGreedIndex,
        TestPortfolioOptimizer,
        TestTradeLogger,
        TestNewsManager
    ]
    
    for test_case in test_cases:
        print(f"Testing {test_case.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_case)
        runner = unittest.TextTestRunner(verbosity=0)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print(f"✓ {test_case.__name__} passed")
        else:
            print(f"✗ {test_case.__name__} failed")
    
    print("Smoke tests completed!")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run smoke tests
    run_smoke_tests() 
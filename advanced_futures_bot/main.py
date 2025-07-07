"""
advanced_futures_bot ana pipeline
"""
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


def main():
    # 1. Veri çek
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch()
    # 2. Sentiment & event tagging
    sentiment = DeepSeekSentiment()
    sentiment_features = sentiment.analyze(df)
    # 3. Order book
    ob_handler = OrderBookHandler()
    ob_features = ob_handler.process()
    # 4. Ortamı başlat
    env = FuturesEnv(sentiment_features, ob_features)
    # 5. Agent ve policy
    policy = HybridPolicy()
    reward_fn = RewardFunction()
    regime = RegimeSwitcher()
    trainer = ContinualTrainer(env, policy, reward_fn, regime)
    # 6. Hyperopt
    hyperopt = HyperOpt(trainer)
    # 7. Eğitim
    trainer.train()
    # 8. Backtest & execution
    exec_engine = ExecutionEngine()
    exec_engine.run()
    # 9. Hedge opsiyonel
    hedge = HedgeOverlay()
    hedge.apply()
    # 10. Explainable AI
    explainer = ExplainableAI()
    explainer.explain()
    # 11. Logging
    logger = Logger()
    logger.log_all()

if __name__ == "__main__":
    main() 
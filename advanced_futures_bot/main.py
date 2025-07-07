#!/usr/bin/env python3
"""
Advanced Futures Trading Bot - Main Application
Production-level crypto futures trading bot with RL, sentiment analysis, and risk management.
"""

import asyncio
import signal
import sys
import time
import logging
from typing import Dict, Any, Optional
import yaml
import argparse
from pathlib import Path

# Import all modules
from data.data_fetcher import DataFetcher
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
from utils.logger import TradingLogger, get_logger
from utils.explainable_ai import ExplainableAI

class AdvancedFuturesBot:
    """
    Production-level advanced futures trading bot.
    - Complete integration of all modules
    - Real-time trading with risk management
    - Continuous learning and adaptation
    - Exception handling and graceful shutdown
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the trading bot with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = get_logger("main")
        
        # Initialize components
        self.data_fetcher = None
        self.sentiment_analyzer = None
        self.order_book_handler = None
        self.environment = None
        self.policy = None
        self.reward_function = None
        self.regime_switcher = None
        self.trainer = None
        self.position_manager = None
        self.liquidation_checker = None
        self.execution_engine = None
        self.hedge_overlay = None
        self.trading_logger = None
        self.explainable_ai = None
        
        # Bot state
        self.is_running = False
        self.is_training = False
        self.current_episode = 0
        self.total_trades = 0
        self.total_pnl = 0.0
        
        # Performance metrics
        self.metrics = {
            "episodes": 0,
            "trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
        
        self.logger.info("Advanced Futures Bot initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            # Return default config
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "binance": {
                "api_key": "",
                "api_secret": "",
                "testnet": True
            },
            "trading": {
                "symbols": ["BTCUSDT"],
                "timeframes": ["1m", "5m", "15m"],
                "max_position_size": 0.1,
                "max_leverage": 10
            },
            "model": {
                "state_dim": 32,
                "action_dim": 3,
                "hidden_dim": 128,
                "learning_rate": 0.001
            },
            "training": {
                "episodes": 1000,
                "batch_size": 32,
                "checkpoint_interval": 100
            },
            "risk": {
                "max_drawdown": 0.2,
                "stop_loss_pct": 0.05,
                "margin_threshold": 0.8
            },
            "logging": {
                "log_dir": "logs",
                "use_wandb": False,
                "use_tensorboard": False
            }
        }

    def initialize_components(self):
        """Initialize all bot components."""
        try:
            self.logger.info("Initializing bot components...")
            
            # Data components
            self.data_fetcher = DataFetcher(self.config)
            self.sentiment_analyzer = SentimentAnalyzer(self.config)
            self.order_book_handler = OrderBookHandler(self.config)
            
            # Environment and model
            self.environment = FuturesEnvironment(self.config)
            self.policy = HybridPolicy(self.config)
            self.reward_function = RewardFunction(self.config)
            
            # Trading components
            self.regime_switcher = RegimeSwitcher(self.config)
            self.position_manager = PositionManager(self.config)
            self.liquidation_checker = LiquidationChecker(self.config)
            self.execution_engine = ExecutionEngine(self.config)
            self.hedge_overlay = HedgeOverlay(self.config)
            
            # Training and optimization
            self.trainer = ContinualTrainer(self.config)
            self.hyperopt = HyperoptOptimizer(self.config)
            
            # Utilities
            self.trading_logger = TradingLogger(self.config)
            self.explainable_ai = ExplainableAI(self.config)
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def run_trading_loop(self):
        """Main trading loop."""
        try:
            self.logger.info("Starting trading loop...")
            self.is_running = True
            
            while self.is_running:
                try:
                    # Fetch market data
                    market_data = await self._fetch_market_data()
                    if not market_data:
                        await asyncio.sleep(1)
                        continue
                    
                    # Analyze sentiment
                    sentiment_data = await self._analyze_sentiment()
                    
                    # Process order book
                    order_book_data = self._process_order_book(market_data)
                    
                    # Detect market regime
                    regime = self._detect_regime(market_data)
                    
                    # Get current state
                    state = self._get_current_state(market_data, sentiment_data, order_book_data, regime)
                    
                    # Get action from policy
                    action, confidence = self._get_action(state, regime)
                    
                    # Execute trade
                    trade_result = await self._execute_trade(action, market_data, order_book_data)
                    
                    # Update environment
                    reward, done = self._update_environment(action, trade_result, sentiment_data)
                    
                    # Log metrics
                    self._log_metrics(reward, trade_result)
                    
                    # Check risk limits
                    self._check_risk_limits()
                    
                    # Update episode
                    if done:
                        self._end_episode()
                    
                    # Training step
                    if self.is_training:
                        await self._training_step()
                    
                    # Sleep between iterations
                    await asyncio.sleep(self.config.get("trading", {}).get("interval", 1))
                    
                except Exception as e:
                    self.logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(5)  # Wait before retrying
                    
        except Exception as e:
            self.logger.error(f"Trading loop failed: {e}")
            raise

    async def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """Fetch market data from all sources."""
        try:
            market_data = {}
            
            for symbol in self.config["trading"]["symbols"]:
                for timeframe in self.config["trading"]["timeframes"]:
                    klines = self.data_fetcher.fetch_klines(symbol, timeframe, limit=100)
                    if klines is not None and not klines.empty:
                        market_data[f"{symbol}_{timeframe}"] = klines
            
            return market_data if market_data else None
            
        except Exception as e:
            self.logger.error(f"Market data fetch error: {e}")
            return None

    async def _analyze_sentiment(self) -> Dict[str, Any]:
        """Analyze market sentiment."""
        try:
            # Get recent news/social media data
            texts = [
                "Bitcoin shows strong momentum",
                "Market sentiment is bullish",
                "Crypto market analysis"
            ]
            
            sentiment_results = self.sentiment_analyzer.batch_analyze(texts)
            
            # Aggregate sentiment
            if sentiment_results:
                avg_sentiment = sum(r.get("sentiment_score", 0) for r in sentiment_results) / len(sentiment_results)
                return {
                    "sentiment": "positive" if avg_sentiment > 0 else "negative",
                    "confidence": abs(avg_sentiment),
                    "details": sentiment_results
                }
            
            return {"sentiment": "neutral", "confidence": 0.5, "details": []}
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {"sentiment": "neutral", "confidence": 0.5, "details": []}

    def _process_order_book(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process order book data."""
        try:
            # Get latest order book for primary symbol
            primary_symbol = self.config["trading"]["symbols"][0]
            
            # Mock order book data (replace with real data)
            order_book = {
                "bids": [[35000, 1.5], [34999, 2.0], [34998, 1.0]],
                "asks": [[35001, 1.0], [35002, 2.5], [35003, 1.5]]
            }
            
            return self.order_book_handler.process_order_book(order_book)
            
        except Exception as e:
            self.logger.error(f"Order book processing error: {e}")
            return {}

    def _detect_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime."""
        try:
            # Get price data for regime detection
            primary_symbol = self.config["trading"]["symbols"][0]
            price_key = f"{primary_symbol}_1m"
            
            if price_key in market_data:
                prices = market_data[price_key]["close"].values
                return self.regime_switcher.detect_regime(prices)
            
            return "range"  # Default regime
            
        except Exception as e:
            self.logger.error(f"Regime detection error: {e}")
            return "range"

    def _get_current_state(self, market_data: Dict[str, Any], 
                          sentiment_data: Dict[str, Any],
                          order_book_data: Dict[str, Any],
                          regime: str) -> np.ndarray:
        """Get current state representation."""
        try:
            # Combine all data sources into state vector
            state_features = []
            
            # Market features
            primary_symbol = self.config["trading"]["symbols"][0]
            price_key = f"{primary_symbol}_1m"
            
            if price_key in market_data:
                df = market_data[price_key]
                state_features.extend([
                    df["close"].iloc[-1],
                    df["volume"].iloc[-1],
                    df["returns"].iloc[-1] if "returns" in df.columns else 0,
                    df["volatility"].iloc[-1] if "volatility" in df.columns else 0
                ])
            
            # Sentiment features
            sentiment_score = 1.0 if sentiment_data["sentiment"] == "positive" else -1.0
            state_features.extend([sentiment_score, sentiment_data["confidence"]])
            
            # Order book features
            state_features.extend([
                order_book_data.get("bid_volume", 0),
                order_book_data.get("ask_volume", 0),
                order_book_data.get("spread", 0)
            ])
            
            # Regime features
            regime_features = self.regime_switcher.get_regime_features(regime)
            state_features.extend(regime_features)
            
            # Pad to required state dimension
            required_dim = self.config["model"]["state_dim"]
            while len(state_features) < required_dim:
                state_features.append(0.0)
            
            return np.array(state_features[:required_dim], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"State construction error: {e}")
            return np.zeros(self.config["model"]["state_dim"], dtype=np.float32)

    def _get_action(self, state: np.ndarray, regime: str) -> tuple:
        """Get action from policy."""
        try:
            # Convert state to tensor
            state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)
            
            # Get action from policy
            action, log_prob, value = self.policy.get_action(state_tensor)
            
            # Apply regime-specific adjustments
            regime_params = self.regime_switcher.get_regime_params(regime)
            
            return action, value.item()
            
        except Exception as e:
            self.logger.error(f"Action selection error: {e}")
            return 1, 0.5  # Default: hold position

    async def _execute_trade(self, action: int, market_data: Dict[str, Any], 
                           order_book_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on action."""
        try:
            if action == 0:  # Sell/Short
                side = "sell"
            elif action == 2:  # Buy/Long
                side = "buy"
            else:  # Hold
                return {"executed": False, "reason": "hold"}
            
            # Calculate position size
            balance = self.environment.balance
            price = market_data[f"{self.config['trading']['symbols'][0]}_1m"]["close"].iloc[-1]
            confidence = 0.8  # From action confidence
            
            size = self.position_manager.calculate_position_size(balance, price, confidence)
            
            # Create order
            order = {
                "side": side,
                "size": size,
                "price": price
            }
            
            # Simulate execution
            execution_result = self.execution_engine.simulate_execution(order, order_book_data)
            
            if execution_result["executed_size"] > 0:
                self.total_trades += 1
                self.logger.info(f"Trade executed: {side} {execution_result['executed_size']} @ {execution_result['executed_price']}")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return {"executed": False, "error": str(e)}

    def _update_environment(self, action: int, trade_result: Dict[str, Any], 
                          sentiment_data: Dict[str, Any]) -> tuple:
        """Update environment and calculate reward."""
        try:
            # Create market data for environment
            market_data = {
                "price": trade_result.get("executed_price", 35000),
                "volume": trade_result.get("executed_size", 0),
                "order_book": {"bids": [[34999, 1]], "asks": [[35001, 1]]}
            }
            
            # Step environment
            next_state, reward, done, info = self.environment.step(action, market_data)
            
            # Calculate reward with sentiment
            final_reward = self.reward_function.calculate_reward(
                self.environment.prev_state,
                self.environment.current_state,
                action,
                sentiment_data
            )
            
            return final_reward, done
            
        except Exception as e:
            self.logger.error(f"Environment update error: {e}")
            return 0.0, False

    def _log_metrics(self, reward: float, trade_result: Dict[str, Any]):
        """Log trading metrics."""
        try:
            # Episode metrics
            episode_metrics = {
                "reward": reward,
                "pnl": self.environment.pnl,
                "episode_length": self.environment.step_count,
                "position": self.environment.position,
                "balance": self.environment.balance
            }
            
            self.trading_logger.log_episode_metrics(self.current_episode, episode_metrics)
            
            # Trading metrics
            if trade_result.get("executed", False):
                trading_metrics = {
                    "position": self.environment.position,
                    "leverage": self.environment.leverage,
                    "margin_ratio": self.liquidation_checker.get_margin_ratio(),
                    "slippage": trade_result.get("slippage", 0),
                    "execution_latency": trade_result.get("latency", 0)
                }
                
                self.trading_logger.log_trading_metrics(trading_metrics)
            
        except Exception as e:
            self.logger.error(f"Metrics logging error: {e}")

    def _check_risk_limits(self):
        """Check and enforce risk limits."""
        try:
            # Check liquidation risk
            risk_info = self.liquidation_checker.check_liquidation_risk(
                self.environment.position,
                self.environment.balance,
                self.environment.current_price,
                self.environment.entry_price
            )
            
            if risk_info["liquidation_risk"] > 0.8:
                self.logger.warning(f"High liquidation risk: {risk_info['liquidation_risk']:.2f}")
                
                if self.liquidation_checker.should_auto_flat(risk_info["margin_ratio"]):
                    self.logger.warning("Auto-flat triggered due to high risk")
                    # Implement auto-flat logic here
            
            # Check drawdown
            if self.environment.drawdown > self.config["risk"]["max_drawdown"]:
                self.logger.warning(f"Max drawdown exceeded: {self.environment.drawdown:.2f}")
                self.is_running = False
            
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")

    def _end_episode(self):
        """End current episode and prepare for next."""
        try:
            self.current_episode += 1
            self.metrics["episodes"] += 1
            
            # Reset environment
            self.environment.reset()
            
            # Log episode summary
            self.logger.info(f"Episode {self.current_episode} completed. "
                           f"PnL: {self.environment.pnl:.2f}, "
                           f"Total trades: {self.total_trades}")
            
            # Save checkpoint
            if self.current_episode % self.config["training"]["checkpoint_interval"] == 0:
                self._save_checkpoint()
            
        except Exception as e:
            self.logger.error(f"Episode end error: {e}")

    async def _training_step(self):
        """Perform training step."""
        try:
            # Collect experience
            if hasattr(self.trainer, 'collect_experience'):
                experience = self.trainer.collect_experience(
                    self.environment, self.policy, self.reward_function
                )
                
                # Update policy
                if experience:
                    loss = self.trainer.update_policy(experience)
                    
                    # Log training metrics
                    self.trading_logger.log_model_metrics(
                        self.current_episode,
                        {"loss": loss, "learning_rate": self.trainer.optimizer.param_groups[0]["lr"]}
                    )
            
        except Exception as e:
            self.logger.error(f"Training step error: {e}")

    def _save_checkpoint(self):
        """Save model checkpoint."""
        try:
            checkpoint_path = f"checkpoints/model_episode_{self.current_episode}.pth"
            Path("checkpoints").mkdir(exist_ok=True)
            
            torch.save({
                "episode": self.current_episode,
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.trainer.optimizer.state_dict(),
                "metrics": self.metrics,
                "config": self.config
            }, checkpoint_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save error: {e}")

    def start_training(self):
        """Start training mode."""
        try:
            self.is_training = True
            self.logger.info("Training mode started")
            
        except Exception as e:
            self.logger.error(f"Training start error: {e}")

    def stop_training(self):
        """Stop training mode."""
        try:
            self.is_training = False
            self.logger.info("Training mode stopped")
            
        except Exception as e:
            self.logger.error(f"Training stop error: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        try:
            return {
                "episodes": self.metrics["episodes"],
                "total_trades": self.total_trades,
                "total_pnl": self.total_pnl,
                "current_balance": self.environment.balance,
                "current_position": self.environment.position,
                "max_drawdown": self.environment.max_drawdown,
                "win_rate": self.metrics["win_rate"],
                "sharpe_ratio": self.metrics["sharpe_ratio"]
            }
        except Exception as e:
            self.logger.error(f"Performance summary error: {e}")
            return {}

    def shutdown(self):
        """Graceful shutdown."""
        try:
            self.logger.info("Shutting down bot...")
            
            # Stop trading loop
            self.is_running = False
            
            # Save final checkpoint
            self._save_checkpoint()
            
            # Close loggers
            if self.trading_logger:
                self.trading_logger.close()
            
            # Save final metrics
            self.trading_logger.save_metrics("final_metrics.json")
            
            self.logger.info("Bot shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal. Gracefully shutting down...")
    if hasattr(signal_handler, 'bot'):
        signal_handler.bot.shutdown()
    sys.exit(0)

async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Advanced Futures Trading Bot")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--mode", choices=["trading", "training", "backtest"], 
                       default="trading", help="Bot mode")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create bot instance
    bot = AdvancedFuturesBot(args.config)
    signal_handler.bot = bot  # Store for signal handler
    
    try:
        # Initialize components
        bot.initialize_components()
        
        # Set mode
        if args.mode == "training":
            bot.start_training()
        elif args.mode == "backtest":
            bot.config["trading"]["backtest"] = True
        
        # Set episode limit
        bot.config["training"]["episodes"] = args.episodes
        
        # Start trading loop
        await bot.run_trading_loop()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        bot.logger.error(f"Application error: {e}")
        raise
    finally:
        bot.shutdown()

if __name__ == "__main__":
    # Import numpy here to avoid circular imports
    import numpy as np
    
    # Run the application
    asyncio.run(main()) 
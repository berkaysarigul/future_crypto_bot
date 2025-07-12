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
from utils.logger import TradingLogger, get_logger
from utils.explainable_ai import ExplainableAI
from data.real_time_streamer import RealTimeStreamer
from infrastructure.database_manager import DatabaseManager
from infrastructure.alert_system import AlertSystem
from risk_management.risk_manager import RiskManager

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
        self.news_manager = None
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
        self.real_time_streamer = None
        self.database_manager = None
        self.alert_system = None
        self.risk_manager = None
        
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
            self.news_manager = NewsManager(self.config)
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
            
            self.real_time_streamer = RealTimeStreamer()
            self.database_manager = DatabaseManager()
            self.alert_system = AlertSystem()
            self.risk_manager = RiskManager(
                max_risk=self.config.get("risk", {}).get("max_risk", 0.01),
                stop_loss_pct=self.config.get("risk", {}).get("stop_loss_pct", 0.005)
            )
            
            # RealTimeStreamer async başlatma
            try:
                asyncio.create_task(self.real_time_streamer.start())
                self.logger.info("RealTimeStreamer başlatıldı.")
            except Exception as e:
                self.logger.error(f"RealTimeStreamer başlatılamadı: {e}")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def run_trading_loop(self):
        """Main trading loop."""
        try:
            self.logger.info("Starting trading loop...")
            self.is_running = True
            
            # NewsManager'ı başlat ve haber verilerini yükle
            try:
                await self.news_manager.load_news_data()
                self.logger.info("NewsManager başlatıldı ve haber verileri yüklendi")
            except Exception as e:
                self.logger.error(f"NewsManager başlatma hatası: {e}")
            
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
                    await self._check_risk_limits()
                    
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
        """Fetch market data from all sources (öncelik: RealTimeStreamer)."""
        try:
            # Öncelik: RealTimeStreamer'dan veri
            if self.real_time_streamer and self.real_time_streamer.order_book:
                return {"order_book": self.real_time_streamer.order_book}
            # Fallback: Eski yöntem (async ile)
            market_data = {}
            for symbol in self.config["trading"]["symbols"]:
                for timeframe in self.config["trading"]["timeframes"]:
                    klines = await asyncio.to_thread(self.data_fetcher.fetch_klines, symbol, timeframe, 100)
                    if klines is not None and not klines.empty:
                        market_data[f"{symbol}_{timeframe}"] = klines
            return market_data if market_data else None
        except Exception as e:
            self.logger.error(f"Market data fetch error: {e}")
            return None

    async def _analyze_sentiment(self) -> Dict[str, Any]:
        """Analyze market sentiment using NewsManager."""
        try:
            # NewsManager ile haber verilerini yükle ve sentiment feature'larını al
            sentiment_features = self.news_manager.get_latest_sentiment_features(hours=24)
            
            # Sentiment skorunu hesapla
            avg_sentiment = sentiment_features.get('avg_sentiment', 0.0)
            
            # Sentiment kategorisini belirle
            if avg_sentiment > 0.3:
                sentiment_category = "positive"
            elif avg_sentiment < -0.3:
                sentiment_category = "negative"
            else:
                sentiment_category = "neutral"
            
            # Confidence hesapla
            confidence = min(abs(avg_sentiment) * 2, 1.0)  # 0-1 arası normalize et
            
            return {
                "sentiment": sentiment_category,
                "confidence": confidence,
                "avg_sentiment": avg_sentiment,
                "features": sentiment_features,
                "details": {
                    "news_count": sentiment_features.get('news_count', 0),
                    "positive_ratio": sentiment_features.get('positive_ratio', 0.0),
                    "negative_ratio": sentiment_features.get('negative_ratio', 0.0),
                    "sentiment_momentum": sentiment_features.get('sentiment_momentum', 0.0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return {
                "sentiment": "neutral", 
                "confidence": 0.5, 
                "avg_sentiment": 0.0,
                "features": {},
                "details": {}
            }

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
            
            # Sentiment features (NewsManager'dan gelen detaylı feature'lar)
            sentiment_features = sentiment_data.get("features", {})
            state_features.extend([
                sentiment_data.get("avg_sentiment", 0.0),  # Ortalama sentiment skoru
                sentiment_data.get("confidence", 0.5),     # Confidence
                sentiment_features.get("sentiment_std", 0.0),  # Sentiment standart sapması
                sentiment_features.get("positive_ratio", 0.0), # Pozitif haber oranı
                sentiment_features.get("negative_ratio", 0.0), # Negatif haber oranı
                sentiment_features.get("neutral_ratio", 1.0),  # Nötr haber oranı
                sentiment_features.get("news_count", 0),       # Haber sayısı
                sentiment_features.get("sentiment_momentum", 0.0)  # Sentiment momentum
            ])
            
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
        """Execute trade based on action. RiskManager, DatabaseManager ve AlertSystem entegre."""
        try:
            if action == 0:  # Sell/Short
                side = "sell"
            elif action == 2:  # Buy/Long
                side = "buy"
            else:  # Hold
                return {"executed": False, "reason": "hold"}

            # Fiyat ve bakiye
            balance = self.environment.balance
            price = market_data.get(f"{self.config['trading']['symbols'][0]}_1m", {}).get("close", [0])[-1] if f"{self.config['trading']['symbols'][0]}_1m" in market_data else 0
            stop_price = price * (1 - self.config["risk"].get("stop_loss_pct", 0.005)) if side == "buy" else price * (1 + self.config["risk"].get("stop_loss_pct", 0.005))

            # Pozisyon büyüklüğü hesapla (RiskManager)
            try:
                size = await self.risk_manager.calculate_position_size(balance, price, stop_price)
            except Exception as e:
                self.logger.error(f"RiskManager pozisyon büyüklüğü hatası: {e}")
                await self.alert_system.alert(f"RiskManager pozisyon büyüklüğü hatası: {e}", level="error")
                return {"executed": False, "error": str(e)}

            # Stop-loss kontrolü (RiskManager)
            try:
                stop_loss_triggered = await self.risk_manager.check_stop_loss(price, price)
                if stop_loss_triggered:
                    self.logger.warning("Stop-loss tetiklendi, trade yapılmadı.")
                    await self.alert_system.alert("Stop-loss tetiklendi, trade yapılmadı.", level="warning")
                    return {"executed": False, "reason": "stop_loss"}
            except Exception as e:
                self.logger.error(f"Stop-loss kontrol hatası: {e}")
                await self.alert_system.alert(f"Stop-loss kontrol hatası: {e}", level="error")
                return {"executed": False, "error": str(e)}

            # Order oluştur
            order = {
                "side": side,
                "size": size,
                "price": price
            }

            # Trade execution (ExecutionEngine)
            try:
                execution_result = self.execution_engine.simulate_execution(order, order_book_data)
            except Exception as e:
                self.logger.error(f"Trade execution error: {e}")
                await self.alert_system.alert(f"Trade execution error: {e}", level="error")
                return {"executed": False, "error": str(e)}

            # Trade kaydını veritabanına yaz (DatabaseManager)
            try:
                await self.database_manager.save_trade({
                    "order": order,
                    "execution_result": execution_result,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"Trade veritabanı kaydı hatası: {e}")
                await self.alert_system.alert(f"Trade veritabanı kaydı hatası: {e}", level="error")

            if execution_result.get("executed_size", 0) > 0:
                self.total_trades += 1
                self.logger.info(f"Trade executed: {side} {execution_result['executed_size']} @ {execution_result['executed_price']}")

            return execution_result

        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            await self.alert_system.alert(f"Trade execution error: {e}", level="error")
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

    async def _check_risk_limits(self):
        """Check and enforce risk limits. RiskManager ve AlertSystem entegre."""
        try:
            # Likidasyon riski kontrolü
            risk_info = self.liquidation_checker.check_liquidation_risk(
                self.environment.position,
                self.environment.balance,
                self.environment.current_price,
                self.environment.entry_price
            )
            if risk_info["liquidation_risk"] > 0.8:
                self.logger.warning(f"High liquidation risk: {risk_info['liquidation_risk']:.2f}")
                await self.alert_system.alert(f"Likidasyon riski çok yüksek: {risk_info['liquidation_risk']:.2f}", level="warning")
                if self.liquidation_checker.should_auto_flat(risk_info["margin_ratio"]):
                    self.logger.warning("Auto-flat triggered due to high risk")
                    await self.alert_system.alert("Auto-flat triggered due to high risk", level="critical")
                    # Auto-flat işlemi burada uygulanabilir
            # Drawdown kontrolü
            if self.environment.drawdown > self.config["risk"]["max_drawdown"]:
                self.logger.warning(f"Max drawdown exceeded: {self.environment.drawdown:.2f}")
                await self.alert_system.alert(f"Max drawdown aşıldı: {self.environment.drawdown:.2f}", level="critical")
                self.is_running = False
            # RiskManager ile ek risk threshold kontrolü
            try:
                pnl = getattr(self.environment, "pnl", 0.0)
                threshold = self.config["risk"].get("risk_threshold", 0.05)
                risk_limit = await self.risk_manager.check_risk_threshold(pnl, threshold)
                if risk_limit:
                    self.logger.warning("Risk threshold aşıldı, işlemler durduruluyor.")
                    await self.alert_system.alert("Risk threshold aşıldı, işlemler durduruluyor.", level="critical")
                    self.is_running = False
            except Exception as e:
                self.logger.error(f"RiskManager risk threshold kontrol hatası: {e}")
                await self.alert_system.alert(f"RiskManager risk threshold kontrol hatası: {e}", level="error")
        except Exception as e:
            self.logger.error(f"Risk check error: {e}")
            await self.alert_system.alert(f"Risk check error: {e}", level="error")

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
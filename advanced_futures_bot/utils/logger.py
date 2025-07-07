import logging
import json
import time
from typing import Any, Dict, Optional, List
from datetime import datetime
import os
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not available, using local logging only")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available")

class TradingLogger:
    """
    Production-level logging system with WandB, TensorBoard, Grafana support.
    - Episode metrics, trading metrics, model metrics
    - Exception handling, file rotation, structured logging
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.log_dir = self.config.get("log_dir", "logs")
        self.wandb_project = self.config.get("wandb_project", "crypto-trading-bot")
        self.wandb_entity = self.config.get("wandb_entity", None)
        self.tensorboard_dir = self.config.get("tensorboard_dir", "runs")
        self.log_level = self.config.get("log_level", "INFO")
        
        # Initialize logging
        self._setup_logging()
        self.wandb_run = None
        self.tensorboard_writer = None
        
        # Metrics storage
        self.episode_metrics = []
        self.trading_metrics = []
        self.model_metrics = []
        
        # Initialize external loggers
        self._init_wandb()
        self._init_tensorboard()
        
        logger.info("TradingLogger initialized successfully")

    def _setup_logging(self):
        """Setup local file logging with rotation."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create formatters
            detailed_formatter = logging.Formatter(
                '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            simple_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)s - %(message)s'
            )
            
            # File handler with rotation
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                os.path.join(self.log_dir, "trading_bot.log"),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, self.log_level))
            file_handler.setFormatter(detailed_formatter)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            
            # Setup root logger
            logging.basicConfig(
                level=getattr(logging, self.log_level),
                handlers=[file_handler, console_handler],
                format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
            )
            
            # Create module loggers
            self.loggers = {
                "main": logging.getLogger("Main"),
                "trading": logging.getLogger("Trading"),
                "model": logging.getLogger("Model"),
                "data": logging.getLogger("Data"),
                "risk": logging.getLogger("Risk")
            }
            
        except Exception as e:
            print(f"Logging setup error: {e}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)

    def _init_wandb(self):
        """Initialize WandB logging."""
        try:
            if WANDB_AVAILABLE and self.config.get("use_wandb", False):
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    entity=self.wandb_entity,
                    config=self.config,
                    name=f"trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                self.loggers["main"].info("WandB initialized successfully")
            else:
                self.loggers["main"].info("WandB disabled or not available")
        except Exception as e:
            self.loggers["main"].error(f"WandB initialization error: {e}")

    def _init_tensorboard(self):
        """Initialize TensorBoard logging."""
        try:
            if TENSORBOARD_AVAILABLE and self.config.get("use_tensorboard", False):
                os.makedirs(self.tensorboard_dir, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(
                    log_dir=os.path.join(self.tensorboard_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
                )
                self.loggers["main"].info("TensorBoard initialized successfully")
            else:
                self.loggers["main"].info("TensorBoard disabled or not available")
        except Exception as e:
            self.loggers["main"].error(f"TensorBoard initialization error: {e}")

    def log_episode_metrics(self, episode: int, metrics: Dict[str, Any]):
        """Log episode-level metrics."""
        try:
            timestamp = time.time()
            episode_data = {
                "episode": episode,
                "timestamp": timestamp,
                **metrics
            }
            self.episode_metrics.append(episode_data)
            
            # Log to different outputs
            self.loggers["main"].info(f"Episode {episode} metrics: {json.dumps(metrics, indent=2)}")
            
            if self.wandb_run:
                wandb.log({"episode": episode, **metrics})
            
            if self.tensorboard_writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f"episode/{key}", value, episode)
            
        except Exception as e:
            self.loggers["main"].error(f"Episode metrics logging error: {e}")

    def log_trading_metrics(self, metrics: Dict[str, Any]):
        """Log trading-specific metrics."""
        try:
            timestamp = time.time()
            trading_data = {
                "timestamp": timestamp,
                **metrics
            }
            self.trading_metrics.append(trading_data)
            
            # Log to different outputs
            self.loggers["trading"].info(f"Trading metrics: {json.dumps(metrics, indent=2)}")
            
            if self.wandb_run:
                wandb.log(metrics)
            
            if self.tensorboard_writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f"trading/{key}", value, int(timestamp))
            
        except Exception as e:
            self.loggers["trading"].error(f"Trading metrics logging error: {e}")

    def log_model_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log model training metrics."""
        try:
            timestamp = time.time()
            model_data = {
                "step": step,
                "timestamp": timestamp,
                **metrics
            }
            self.model_metrics.append(model_data)
            
            # Log to different outputs
            self.loggers["model"].info(f"Model step {step} metrics: {json.dumps(metrics, indent=2)}")
            
            if self.wandb_run:
                wandb.log({"step": step, **metrics})
            
            if self.tensorboard_writer:
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        self.tensorboard_writer.add_scalar(f"model/{key}", value, step)
            
        except Exception as e:
            self.loggers["model"].error(f"Model metrics logging error: {e}")

    def log_error(self, error: Exception, context: str = ""):
        """Log errors with context."""
        try:
            error_data = {
                "timestamp": time.time(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            }
            
            self.loggers["main"].error(f"Error in {context}: {error}")
            
            if self.wandb_run:
                wandb.log({"error": True, "error_message": str(error)})
            
        except Exception as e:
            print(f"Error logging failed: {e}")

    def log_warning(self, message: str, context: str = ""):
        """Log warnings with context."""
        try:
            self.loggers["main"].warning(f"Warning in {context}: {message}")
        except Exception as e:
            print(f"Warning logging failed: {e}")

    def log_info(self, message: str, context: str = ""):
        """Log info messages with context."""
        try:
            self.loggers["main"].info(f"Info in {context}: {message}")
        except Exception as e:
            print(f"Info logging failed: {e}")

    def save_metrics(self, filename: str = None):
        """Save all metrics to JSON file."""
        try:
            if filename is None:
                filename = f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            filepath = os.path.join(self.log_dir, filename)
            
            all_metrics = {
                "episode_metrics": self.episode_metrics,
                "trading_metrics": self.trading_metrics,
                "model_metrics": self.model_metrics,
                "config": self.config,
                "export_time": datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            
            self.loggers["main"].info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            self.loggers["main"].error(f"Metrics save error: {e}")

    def close(self):
        """Close all logging connections."""
        try:
            if self.wandb_run:
                self.wandb_run.finish()
                self.loggers["main"].info("WandB run finished")
            
            if self.tensorboard_writer:
                self.tensorboard_writer.close()
                self.loggers["main"].info("TensorBoard writer closed")
            
            self.loggers["main"].info("TradingLogger closed successfully")
            
        except Exception as e:
            print(f"Logger close error: {e}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of logged metrics."""
        try:
            if not self.episode_metrics:
                return {"error": "No metrics available"}
            
            # Episode metrics summary
            episode_rewards = [m.get("reward", 0) for m in self.episode_metrics]
            episode_lengths = [m.get("episode_length", 0) for m in self.episode_metrics]
            
            # Trading metrics summary
            pnl_values = [m.get("pnl", 0) for m in self.trading_metrics]
            win_rates = [m.get("win_rate", 0) for m in self.trading_metrics]
            
            summary = {
                "total_episodes": len(self.episode_metrics),
                "total_trades": len(self.trading_metrics),
                "avg_reward": np.mean(episode_rewards) if episode_rewards else 0,
                "avg_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
                "total_pnl": sum(pnl_values) if pnl_values else 0,
                "avg_win_rate": np.mean(win_rates) if win_rates else 0,
                "max_reward": max(episode_rewards) if episode_rewards else 0,
                "min_reward": min(episode_rewards) if episode_rewards else 0
            }
            
            return summary
            
        except Exception as e:
            self.loggers["main"].error(f"Summary stats error: {e}")
            return {"error": str(e)}

# Global logger instance
logger = TradingLogger()

def get_logger(name: str = "main"):
    """Get logger instance by name."""
    return logger.loggers.get(name, logger.loggers["main"]) 
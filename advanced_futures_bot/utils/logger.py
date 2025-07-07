from typing import Any, Dict, Optional
import os
import json
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    WandB, TensorBoard, Grafana compatible logging.
    - Episode metrics
    - Trading metrics
    - Model metrics
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.wandb_enabled = self.config.get("wandb", True)
        self.tensorboard_enabled = self.config.get("tensorboard", True)
        self.grafana_enabled = self.config.get("grafana", False)
        self.log_dir = self.config.get("log_dir", "logs")
        
        # Initialize loggers
        self.wandb_run = None
        self.tensorboard_writer = None
        self.grafana_data = []
        
        self._setup_loggers()

    def _setup_loggers(self):
        """Initialize logging backends."""
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # WandB setup
        if self.wandb_enabled:
            try:
                self.wandb_run = wandb.init(
                    project="advanced_futures_bot",
                    config=self.config,
                    name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except Exception as e:
                print(f"WandB setup failed: {e}")
                self.wandb_enabled = False
        
        # TensorBoard setup
        if self.tensorboard_enabled:
            try:
                self.tensorboard_writer = SummaryWriter(
                    log_dir=os.path.join(self.log_dir, "tensorboard")
                )
            except Exception as e:
                print(f"TensorBoard setup failed: {e}")
                self.tensorboard_enabled = False

    def log_episode(self, episode: int, reward: float, pnl: float, regime: str):
        """Log episode metrics."""
        metrics = {
            "episode": episode,
            "reward": reward,
            "pnl": pnl,
            "regime": regime,
            "timestamp": datetime.now().isoformat()
        }
        
        # WandB
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.log(metrics)
        
        # TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar("Episode/Reward", reward, episode)
            self.tensorboard_writer.add_scalar("Episode/PnL", pnl, episode)
            self.tensorboard_writer.add_text("Episode/Regime", regime, episode)
        
        # Grafana (JSON format)
        if self.grafana_enabled:
            self.grafana_data.append(metrics)

    def log_trading(self, 
                   position: float,
                   leverage: float,
                   margin_ratio: float,
                   slippage: float,
                   execution_type: str):
        """Log trading metrics."""
        metrics = {
            "position": position,
            "leverage": leverage,
            "margin_ratio": margin_ratio,
            "slippage": slippage,
            "execution_type": execution_type,
            "timestamp": datetime.now().isoformat()
        }
        
        # WandB
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.log(metrics)
        
        # TensorBoard
        if self.tensorboard_writer:
            step = len(self.grafana_data) if self.grafana_data else 0
            self.tensorboard_writer.add_scalar("Trading/Position", position, step)
            self.tensorboard_writer.add_scalar("Trading/Leverage", leverage, step)
            self.tensorboard_writer.add_scalar("Trading/MarginRatio", margin_ratio, step)
            self.tensorboard_writer.add_scalar("Trading/Slippage", slippage, step)
        
        # Grafana
        if self.grafana_enabled:
            self.grafana_data.append(metrics)

    def log_model(self, 
                 learning_rate: float,
                 entropy: float,
                 value_loss: float,
                 policy_loss: float):
        """Log model training metrics."""
        metrics = {
            "learning_rate": learning_rate,
            "entropy": entropy,
            "value_loss": value_loss,
            "policy_loss": policy_loss,
            "timestamp": datetime.now().isoformat()
        }
        
        # WandB
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.log(metrics)
        
        # TensorBoard
        if self.tensorboard_writer:
            step = len(self.grafana_data) if self.grafana_data else 0
            self.tensorboard_writer.add_scalar("Model/LearningRate", learning_rate, step)
            self.tensorboard_writer.add_scalar("Model/Entropy", entropy, step)
            self.tensorboard_writer.add_scalar("Model/ValueLoss", value_loss, step)
            self.tensorboard_writer.add_scalar("Model/PolicyLoss", policy_loss, step)

    def log_all(self):
        """Log all accumulated data."""
        # Save Grafana data
        if self.grafana_enabled and self.grafana_data:
            grafana_file = os.path.join(self.log_dir, "grafana_data.json")
            with open(grafana_file, 'w') as f:
                json.dump(self.grafana_data, f, indent=2)
        
        # Close TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Finish WandB run
        if self.wandb_enabled and self.wandb_run:
            self.wandb_run.finish() 
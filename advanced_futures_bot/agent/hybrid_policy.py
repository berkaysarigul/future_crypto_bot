import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Tuple, Dict
import logging
import os

logger = logging.getLogger("HybridPolicy")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class HybridPolicy(nn.Module):
    """
    Production-level PPO + LSTM + Transformer hybrid policy.
    - Policy & value head
    - PPO loss, optimizer, checkpoint, logging
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_actions: int = 3, regime_dim: int = 4, lr: float = 3e-4, device: str = "cpu"):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + regime_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.device = torch.device(device)
        self.to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.clip_grad_norm = 0.5

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None, hidden: Optional[Any] = None) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        x: [batch, seq, input_dim]
        regime: [batch, seq, regime_dim] veya None
        """
        if regime is not None:
            x = torch.cat([x, regime], dim=-1)
        x = self.input_proj(x)
        # LSTM
        if hidden is None:
            x, hidden = self.lstm(x)
        else:
            x, hidden = self.lstm(x, hidden)
        # Transformer
        x = self.transformer(x)
        # Son zaman adımını al
        x_last = x[:, -1, :]
        logits = self.policy_head(x_last)
        value = self.value_head(x_last)
        return logits, value.squeeze(-1), hidden

    def act(self, state: torch.Tensor, regime: Optional[torch.Tensor] = None, hidden: Optional[Any] = None, deterministic: bool = False) -> Tuple[int, float, Any]:
        self.eval()
        with torch.no_grad():
            logits, value, hidden = self.forward(state.to(self.device), regime.to(self.device) if regime is not None else None, hidden)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return action.item(), value.item(), hidden

    def compute_ppo_loss(self, batch: Dict[str, torch.Tensor], clip_range: float = 0.2, entropy_coef: float = 0.01, value_coef: float = 0.5) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        PPO loss fonksiyonu (production-level):
        batch: dict with keys: state, regime, action, old_logprob, advantage, return, value
        """
        state = batch["state"].to(self.device)
        regime = batch["regime"].to(self.device)
        action = batch["action"].to(self.device)
        old_logprob = batch["old_logprob"].to(self.device)
        advantage = batch["advantage"].to(self.device)
        returns = batch["return"].to(self.device)
        value = batch["value"].to(self.device)

        logits, new_value, _ = self.forward(state, regime)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_logprob = dist.log_prob(action)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_logprob - old_logprob)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (returns - new_value).pow(2).mean()
        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item()
        }
        return loss, stats

    def update(self, batch: Dict[str, torch.Tensor], clip_range: float = 0.2, entropy_coef: float = 0.01, value_coef: float = 0.5) -> Dict[str, float]:
        self.train()
        self.optimizer.zero_grad()
        loss, stats = self.compute_ppo_loss(batch, clip_range, entropy_coef, value_coef)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        logger.info(f"PPO update: {stats}")
        return stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        logger.info(f"Model loaded from {path}") 
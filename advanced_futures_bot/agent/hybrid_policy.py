import torch
import torch.nn as nn
from typing import Any, Optional

class HybridPolicy(nn.Module):
    """
    PPO + LSTM + Transformer hybrid policy.
    - Girdi: hybrid feature vektörü + regime context
    - Çıktı: action logits, value
    """
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_actions: int = 3, regime_dim: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + regime_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, regime: Optional[torch.Tensor] = None, hidden: Optional[Any] = None):
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

    def act(self, state: torch.Tensor, regime: Optional[torch.Tensor] = None, hidden: Optional[Any] = None):
        self.eval()
        with torch.no_grad():
            logits, value, hidden = self.forward(state, regime, hidden)
            action = torch.argmax(logits, dim=-1)
        return action.item(), value.item(), hidden

    def update(self, *args, **kwargs):
        # PPO update burada implemente edilecek
        pass 
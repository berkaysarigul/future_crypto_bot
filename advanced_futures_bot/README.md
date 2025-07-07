<!-- CURSOR CONTEXT START -->

## 🚀 BTCUSDT Advanced Futures RL Bot + DeepSeek

- **Amaç:** Binance BTCUSDT Futures, PPO + LSTM + Transformer hybrid.
- **Sentiment:** DeepSeek LLM API + event tag.
- **Data:** Order Book depth, OI, Funding Rate, Whale, On-chain.
- **Reward:** Conditional PnL + VaR + funding penalty + liquidation.
- **Execution:** Realistic slippage, TWAP, partial fill.
- **Regime Switch:** Trend/range detect → param auto adjust.
- **Hedge:** Optional spread / perp premium.
- **Explainable AI:** SHAP, LIME.
- **Logging:** WandB, TensorBoard, Grafana.
- **Config:** `config.yaml`
- **Secrets:** `.env` or vault.
- **Tests:** Only Futures, no spot.

**Klasör:**
- data/, env/, agent/, trading/, utils/, tests/, main.py
- Tüm modüller directory_structure ve rules'a sadık.

<!-- CURSOR CONTEXT END -->

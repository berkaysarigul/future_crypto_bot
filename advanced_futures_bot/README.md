<!-- CURSOR CONTEXT START -->

## ⚡ BTCUSDT Advanced Futures RL Bot + DeepSeek LLM

### 🎯 Amaç
- Binance BTCUSDT Futures otomatik RL trading.
- Hybrid PPO + LSTM + Transformer policy.
- DeepSeek LLM API → sentiment & event tag.
- Realistic backtest + live trading + failover + health check.

### 📂 Klasör Yapısı
- config/ → env, trading, model configs
- data/ → fetcher, sentiment, order book, streamer, feature eng.
- env/ → futures_env
- agent/ → hybrid policy, reward, hyperopt, regime switch
- trading/ → position manager, liquidation, execution, hedge
- risk_management/ → VaR, drawdown, position sizing, stress test
- backtest/ → backtester, walk-forward, monte carlo
- models/ → versioning, registry, drift detector
- infrastructure/ → health check, alert, DB, failover
- external_data/ → news, social media, whale, fear/greed
- strategies/ → signal gen, portfolio optimizer, rebalancing
- utils/ → logger, explainable AI, trade logger
- tests/ → full test
- main.py → pipeline: fetch → env → agent → trade → log → monitor

### ✅ Kurallar
- Sadece Futures.
- Action: -1 short, 0 flat, +1 long.
- Sentiment DeepSeek + order book + funding + macro combined.
- Reward shaping: PnL + VaR + risk.
- Drift detect aktif.
- Health + failover + alerts.
- Trade logger trace.
- Logging: WandB, TB, Grafana.
- API keys `.env`.

<!-- CURSOR CONTEXT END -->

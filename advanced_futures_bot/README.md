<!-- CURSOR CONTEXT START -->

## ⚡ BTCUSDT Advanced Futures RL Bot + DeepSeek LLM

### 🎯 Amaç
- Binance BTCUSDT Futures için tam otomatik Reinforcement Learning trading bot.
- Hybrid PPO + LSTM + Transformer policy.
- DeepSeek LLM API → haber/sentiment & event tag.
- Realistic walk-forward backtest + live trading + drift detect + failover + health check.

---

### 📂 Klasör Yapısı
- **config/** → env, trading, model configs, `config_validator.py`
- **data/** → fetcher, sentiment, order book, real_time_streamer, feature_engineering, `data_validator.py`, `news_manager.py`
- **env/** → futures_env
- **agent/** → hybrid policy, reward_function, hyperopt, regime_switcher, continual_trainer
- **trading/** → position_manager, liquidation_checker, execution_engine, hedge_overlay
- **risk_management/** → var_calculator, risk_metrics, position_sizing, risk_manager, stress_testing
- **backtest/** → backtester, performance_analyzer, walk_forward_validator, monte_carlo_simulator
- **models/** → model_manager (save/load/versioning), drift_detector
- **infrastructure/** → health_checker, alert_system, database_manager, failover_handler
- **external_data/** → news_scraper, social_media_monitor, whale_tracker, fear_greed_index
- **strategies/** → signal_generator, portfolio_optimizer, rebalancing_engine, strategy_evaluator
- **utils/** → logger, explainable_ai, trade_logger
- **tests/** → full test coverage: `test_futures_bot.py`
- **main.py** → pipeline: fetch → env → agent → trade → log → monitor

---

### ✅ Kurallar
- Sadece **Futures Logic**: Spot yok.
- Action space: -1 (short) | 0 (flat) | +1 (long)
- Hybrid input: DeepSeek sentiment + L2 order book + funding rates + macro features.
- Reward shaping: PnL + funding fee + VaR + drawdown penalty.
- Drift detect aktif.
- Backtester: Walk-forward, realistic slippage, latency.
- Health check + failover + alerts zorunlu.
- Full trade logger → kim açtı, hangi regime, sinyal kaydı.
- Model versioning + A/B test.
- Tüm secrets `.env` veya vault.
- Async/await uyumlu.
- Exception handling & structured logging: WandB, TensorBoard, Grafana.

---

### 🚀 Phase 1-2-3 Yeni Modüller ve Bağlantılar

✅ `data/real_time_streamer.py` → Binance WebSocket → L2 tick feed → execution_engine.py  
✅ `infrastructure/database_manager.py` → SQLite/PostgreSQL → trade history, feature store  
✅ `infrastructure/alert_system.py` → Telegram/Discord alerts → risk breach, drift detect, failover  
✅ `risk_management/risk_manager.py` → Position sizing, stop-loss automation, VaR monitor  
✅ `backtest/backtester.py` → Walk-forward validation, monte_carlo_simulator  
✅ `models/model_manager.py` → Checkpoint save/load, version control, A/B test  
✅ `analytics/performance_analyzer.py` → Sharpe, Sortino, Drawdown, Win/Loss  
✅ `data/data_validator.py` → Anomaly detect, missing data fix  
✅ `config/config_validator.py` → Param schema validation  
✅ `models/drift_detector.py` → Performance drop detect, threshold alert  
✅ `risk_management/stress_testing.py` → Flash crash, black swan simülasyon  
✅ `infrastructure/failover_handler.py` → API down veya margin breach → pozisyonları flat kapat  
✅ `strategies/strategy_evaluator.py` → Multi-strategy comparison & selection  
✅ `external_data/news_scraper.py` → Güncel haber tarayıcı  
✅ `external_data/social_media_monitor.py` → Twitter, Reddit real-time sentiment  
✅ `external_data/whale_tracker.py` → Whale Alert tracking  
✅ `external_data/fear_greed_index.py` → Piyasa sentiment index input  
✅ `strategies/portfolio_optimizer.py` → Çoklu varlık desteği olursa portföy optimizasyon  
✅ `utils/trade_logger.py` → Tam trade trace → kim, ne zaman, hangi sinyal, PnL, funding  
✅ `data/news_manager.py` → Haber verisi yönetimi, CSV kontrolü, sentiment analizi entegrasyonu

---

### 🆕 Phase 2 Modülleri
- **backtest/backtester.py**: Walk-forward backtest, latency, funding fee, slippage
- **models/model_manager.py**: Model versioning, checkpoint, A/B test
- **analytics/performance_analyzer.py**: Sharpe, Sortino, Drawdown, Win/Loss
- **data/data_validator.py**: Data sanity, missing/duplicate fix
- **config/config_validator.py**: Config schema check, env param validation

---

### ⚙️ .env Kullanımı
Tüm API key'leri ve secret ayarları `.env` dosyasından yönetilir. Örnek: `.env.example` dosyasını doldurun.

---

### 🔄 Pipeline'da Yeri
- **Veri Akışı:** `RealTimeStreamer`, `data_validator`
- **Trade ve Risk Yönetimi:** `RiskManager`, `DatabaseManager`, `FailoverHandler`, `AlertSystem`
- **Strateji & Model:** `HybridPolicy`, `ModelManager`, `DriftDetector`, `StrategyEvaluator`
- **Performans & Monitoring:** `PerformanceAnalyzer`, `TradeLogger`, `Logger`
- **Tüm modüller:** async/await uyumlu, robust logging & exception handling → production grade.
- **Backtest & Model:** `Backtester`, `ModelManager`, `PerformanceAnalyzer`
- **Veri Temizlik & Config:** `DataValidator`, `ConfigValidator`

---

### 🚀 Hızlı Başlangıç
```bash
cp .env.example .env
pip install -r requirements.txt
python main.py --mode trading
daha fazla detay, yapılandırma ve gelişmiş kullanım için config/, README ve modül dökümantasyonlarını inceleyin.

---

## 📰 NewsManager Kullanım Kılavuzu

### 🎯 NewsManager Nedir?
`NewsManager` modülü, haber verilerini yöneten ve sentiment analizi entegrasyonu sağlayan production-grade bir bileşendir. Haber verilerini CSV dosyasından okur, gerekirse yeni haber çeker ve sentiment skorlarını hesaplar.

### 🔧 Nasıl Çağrılır?

#### 1️⃣ Temel Kullanım
```python
from data.news_manager import NewsManager

# NewsManager'ı başlat
news_manager = NewsManager(config={
    "news_sources": ["https://api.coingecko.com/api/v3/news"],
    "sentiment": {
        "api_url": "https://api.deepseek.com/v1/sentiment",
        "batch_size": 8
    },
    "update_interval": 3600  # 1 saat
})

# Haber verilerini yükle
await news_manager.load_news_data()

# Sentiment feature'larını al
sentiment_features = news_manager.get_latest_sentiment_features(hours=24)
```

#### 2️⃣ main.py İçinde RL Observation'a Ekleme
```python
# main.py içinde _analyze_sentiment metodunda:
async def _analyze_sentiment(self) -> Dict[str, Any]:
    try:
        # NewsManager ile sentiment feature'larını al
        sentiment_features = self.news_manager.get_latest_sentiment_features(hours=24)
        
        # Sentiment skorunu hesapla
        avg_sentiment = sentiment_features.get('avg_sentiment', 0.0)
        
        return {
            "sentiment": "positive" if avg_sentiment > 0.3 else "negative" if avg_sentiment < -0.3 else "neutral",
            "confidence": min(abs(avg_sentiment) * 2, 1.0),
            "avg_sentiment": avg_sentiment,
            "features": sentiment_features
        }
    except Exception as e:
        self.logger.error(f"Sentiment analysis error: {e}")
        return {"sentiment": "neutral", "confidence": 0.5, "avg_sentiment": 0.0, "features": {}}

# _get_current_state metodunda sentiment feature'larını RL observation'a ekle:
def _get_current_state(self, market_data, sentiment_data, order_book_data, regime):
    state_features = []
    
    # Market features...
    
    # Sentiment features (NewsManager'dan gelen detaylı feature'lar)
    sentiment_features = sentiment_data.get("features", {})
    state_features.extend([
        sentiment_data.get("avg_sentiment", 0.0),      # Ortalama sentiment skoru
        sentiment_data.get("confidence", 0.5),         # Confidence
        sentiment_features.get("sentiment_std", 0.0),  # Sentiment standart sapması
        sentiment_features.get("positive_ratio", 0.0), # Pozitif haber oranı
        sentiment_features.get("negative_ratio", 0.0), # Negatif haber oranı
        sentiment_features.get("neutral_ratio", 1.0),  # Nötr haber oranı
        sentiment_features.get("news_count", 0),       # Haber sayısı
        sentiment_features.get("sentiment_momentum", 0.0)  # Sentiment momentum
    ])
    
    # Order book features...
    
    return np.array(state_features, dtype=np.float32)
```

### 🧪 Test Snippet Örneği

#### NewsManager Başlatma ve Test
```python
import asyncio
from data.news_manager import NewsManager

async def test_news_manager():
    # NewsManager'ı başlat
    news_manager = NewsManager()
    
    # Haber verilerini yükle
    df = await news_manager.load_news_data()
    print(f"Yüklenen haber sayısı: {len(df)}")
    
    # Sentiment feature'larını al
    features = news_manager.get_latest_sentiment_features(hours=24)
    print("Sentiment Features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
    
    # Haber özetini al
    summary = news_manager.get_news_summary()
    print(f"Haber Özeti: {summary}")
    
    # Verileri yenile
    success = await news_manager.refresh_news_data()
    print(f"Veri yenileme başarılı: {success}")

# Test'i çalıştır
if __name__ == "__main__":
    asyncio.run(test_news_manager())
```

#### Observation Dict'e Ekleme
```python
# RL observation dict'ine sentiment feature'larını ekle
def create_observation_dict(market_data, sentiment_features):
    observation = {
        # Market data
        "price": market_data.get("close", 0),
        "volume": market_data.get("volume", 0),
        "returns": market_data.get("returns", 0),
        
        # Sentiment features (NewsManager'dan)
        "avg_sentiment": sentiment_features.get("avg_sentiment", 0.0),
        "sentiment_confidence": sentiment_features.get("sentiment_std", 0.0),
        "positive_news_ratio": sentiment_features.get("positive_ratio", 0.0),
        "negative_news_ratio": sentiment_features.get("negative_ratio", 0.0),
        "news_count": sentiment_features.get("news_count", 0),
        "sentiment_momentum": sentiment_features.get("sentiment_momentum", 0.0),
        
        # Diğer features...
    }
    return observation

# Kullanım
news_manager = NewsManager()
sentiment_features = news_manager.get_latest_sentiment_features(hours=24)
observation = create_observation_dict(market_data, sentiment_features)
```

### 📊 Sentiment Feature'ları

NewsManager şu sentiment feature'larını sağlar:

- **avg_sentiment**: Ortalama sentiment skoru (-1 ile +1 arası)
- **sentiment_std**: Sentiment standart sapması
- **positive_ratio**: Pozitif haber oranı (0-1 arası)
- **negative_ratio**: Negatif haber oranı (0-1 arası)
- **neutral_ratio**: Nötr haber oranı (0-1 arası)
- **news_count**: Son N saatteki haber sayısı
- **sentiment_momentum**: Sentiment değişim momentumu

### 🔄 Veri Akışı

1. **CSV Kontrolü**: `data/news_data/cryptonews.csv` dosyası kontrol edilir
2. **Sentiment Kontrolü**: Eğer sentiment sütunu eksikse, DeepSeek API ile analiz yapılır
3. **Yeni Veri**: CSV yoksa, news_scraper ile haber çekilir
4. **Feature Extraction**: Son N saatlik verilerden sentiment feature'ları çıkarılır
5. **RL Integration**: Feature'lar RL observation'a eklenir

### ⚠️ Önemli Notlar

- NewsManager async/await uyumludur
- Exception handling ve logging içerir
- CSV dosyası otomatik oluşturulur
- Sentiment analizi DeepSeek API kullanır
- Veri güncelleme interval'i configurable'dır

<!-- CURSOR CONTEXT END -->
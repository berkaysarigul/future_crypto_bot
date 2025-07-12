#!/usr/bin/env python3
"""
News Manager Module
Production-grade haber verisi yönetimi ve sentiment analizi entegrasyonu.
"""

import os
import asyncio
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import aiofiles
import json

from external_data.news_scraper import NewsScraper
from data.sentiment_deepseek import DeepSeekSentiment

class NewsManager:
    """
    Production-grade haber verisi yöneticisi.
    
    Özellikler:
    - CSV dosyası kontrolü ve yönetimi
    - Async haber çekme ve sentiment analizi
    - Otomatik veri güncelleme
    - Exception handling ve logging
    - RL observation için feature extraction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """NewsManager'ı başlat."""
        self.config = config or {}
        self.logger = logging.getLogger("NewsManager")
        
        # Dosya yolları
        self.data_dir = Path("data/news_data")
        self.csv_path = self.data_dir / "cryptonews.csv"
        
        # Haber kaynakları
        self.news_sources = self.config.get("news_sources", [
            "https://api.coingecko.com/api/v3/news",
            "https://cryptonews-api.com/api/v1/news"
        ])
        
        # Sentiment analizi ayarları
        self.sentiment_config = self.config.get("sentiment", {})
        self.update_interval = self.config.get("update_interval", 3600)  # 1 saat
        
        # Bileşenler
        self.news_scraper = NewsScraper(self.news_sources)
        self.sentiment_analyzer = DeepSeekSentiment(config=self.sentiment_config)
        
        # Cache
        self._news_cache = None
        self._last_update = None
        
        # Veri dizinini oluştur
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("NewsManager başlatıldı")
    
    async def load_news_data(self) -> pd.DataFrame:
        """
        Haber verilerini yükle.
        
        Returns:
            DataFrame: Haber verileri ve sentiment skorları
        """
        try:
            self.logger.info("Haber verileri yükleniyor...")
            
            # CSV dosyası var mı kontrol et
            if self.csv_path.exists():
                df = pd.read_csv(self.csv_path)
                self.logger.info(f"CSV dosyası bulundu: {len(df)} kayıt")
                
                # Sentiment sütunu var mı ve dolu mu kontrol et
                if 'sentiment' in df.columns and not df['sentiment'].isna().all():
                    self.logger.info("Sentiment verileri mevcut, güncelleme gerekmiyor")
                    return df
                else:
                    self.logger.info("Sentiment sütunu eksik, güncelleniyor...")
                    return await self._update_sentiment(df)
            else:
                self.logger.info("CSV dosyası bulunamadı, yeni veri oluşturuluyor...")
                return await self._create_news_csv()
                
        except Exception as e:
            self.logger.error(f"Haber verisi yükleme hatası: {e}")
            # Hata durumunda boş DataFrame döndür
            return pd.DataFrame(columns=['title', 'content', 'source', 'timestamp', 'sentiment'])
    
    async def _update_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mevcut haber verilerine sentiment skorları ekle.
        
        Args:
            df: Mevcut haber DataFrame'i
            
        Returns:
            DataFrame: Sentiment skorları eklenmiş DataFrame
        """
        try:
            self.logger.info("Sentiment analizi başlatılıyor...")
            
            # Sentiment sütunu yoksa ekle
            if 'sentiment' not in df.columns:
                df['sentiment'] = None
            
            # Sentiment'i olmayan kayıtları bul
            missing_sentiment = df[df['sentiment'].isna()]
            
            if len(missing_sentiment) == 0:
                self.logger.info("Tüm kayıtların sentiment skoru mevcut")
                return df
            
            self.logger.info(f"{len(missing_sentiment)} kayıt için sentiment analizi yapılıyor...")
            
            # Metinleri hazırla
            texts = []
            for _, row in missing_sentiment.iterrows():
                text = f"{row.get('title', '')} {row.get('content', '')}"
                texts.append(text)
            
            # Sentiment analizi yap
            sentiment_results = self.sentiment_analyzer.analyze(texts)
            
            # Sonuçları DataFrame'e ekle
            sentiment_idx = 0
            for idx, row in missing_sentiment.iterrows():
                if sentiment_idx < len(sentiment_results):
                    result = sentiment_results[sentiment_idx]
                    if 'sentiment' in result and result['sentiment'] is not None:
                        df.loc[idx, 'sentiment'] = result['sentiment']
                    sentiment_idx += 1
            
            # CSV'ye kaydet
            df.to_csv(self.csv_path, index=False)
            self.logger.info("Sentiment analizi tamamlandı ve kaydedildi")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Sentiment güncelleme hatası: {e}")
            return df
    
    async def _create_news_csv(self) -> pd.DataFrame:
        """
        Yeni haber verisi oluştur ve CSV'ye kaydet.
        
        Returns:
            DataFrame: Yeni oluşturulan haber DataFrame'i
        """
        try:
            self.logger.info("Yeni haber verisi oluşturuluyor...")
            
            # Haberleri çek
            news_data = await self.news_scraper.fetch_news()
            
            if not news_data:
                self.logger.warning("Haber verisi çekilemedi, örnek veri oluşturuluyor")
                news_data = self._create_sample_news()
            
            # DataFrame oluştur
            df = pd.DataFrame(news_data)
            
            # Gerekli sütunları ekle
            required_columns = ['title', 'content', 'source', 'timestamp']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Timestamp'i düzelt
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['timestamp'].fillna(datetime.now(), inplace=True)
            
            # Sentiment analizi yap
            df = await self._update_sentiment(df)
            
            # CSV'ye kaydet
            df.to_csv(self.csv_path, index=False)
            self.logger.info(f"Yeni haber CSV'si oluşturuldu: {len(df)} kayıt")
            
            return df
            
        except Exception as e:
            self.logger.error(f"CSV oluşturma hatası: {e}")
            return pd.DataFrame(columns=['title', 'content', 'source', 'timestamp', 'sentiment'])
    
    def _create_sample_news(self) -> List[Dict[str, Any]]:
        """Örnek haber verisi oluştur."""
        sample_news = [
            {
                'title': 'Bitcoin Fiyatı 50,000$ Seviyesini Test Ediyor',
                'content': 'Bitcoin son günlerde güçlü bir yükseliş gösteriyor ve 50,000$ seviyesini test ediyor. Uzmanlar bu seviyenin aşılması durumunda daha yüksek hedeflere ulaşılabileceğini belirtiyor.',
                'source': 'sample',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Ethereum 2.0 Güncellemesi Başarıyla Tamamlandı',
                'content': 'Ethereum ağında beklenen 2.0 güncellemesi başarıyla tamamlandı. Bu güncelleme ile ağ daha hızlı ve güvenli hale geldi.',
                'source': 'sample',
                'timestamp': datetime.now().isoformat()
            },
            {
                'title': 'Kripto Piyasalarında Volatilite Artıyor',
                'content': 'Kripto para piyasalarında volatilite seviyeleri artmaya devam ediyor. Yatırımcılar risk yönetimi konusunda dikkatli olmalı.',
                'source': 'sample',
                'timestamp': datetime.now().isoformat()
            }
        ]
        return sample_news
    
    def get_latest_sentiment_features(self, hours: int = 24) -> Dict[str, float]:
        """
        Son N saatlik haber verilerinden sentiment feature'ları çıkar.
        
        Args:
            hours: Kaç saatlik veri alınacak
            
        Returns:
            Dict: Sentiment feature'ları
        """
        try:
            # CSV'yi yükle
            if not self.csv_path.exists():
                self.logger.warning("CSV dosyası bulunamadı, boş feature'lar döndürülüyor")
                return self._get_empty_features()
            
            df = pd.read_csv(self.csv_path)
            
            if df.empty:
                return self._get_empty_features()
            
            # Timestamp sütununu datetime'a çevir
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Son N saatlik veriyi filtrele
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_news = df[df['timestamp'] >= cutoff_time]
            
            if recent_news.empty:
                self.logger.info(f"Son {hours} saatte haber verisi bulunamadı")
                return self._get_empty_features()
            
            # Sentiment skorlarını hesapla
            sentiment_scores = recent_news['sentiment'].dropna()
            
            if sentiment_scores.empty:
                return self._get_empty_features()
            
            # Feature'ları hesapla
            features = {
                'avg_sentiment': float(sentiment_scores.mean()),
                'sentiment_std': float(sentiment_scores.std()),
                'positive_ratio': float((sentiment_scores > 0.5).mean()),
                'negative_ratio': float((sentiment_scores < -0.5).mean()),
                'neutral_ratio': float(((sentiment_scores >= -0.5) & (sentiment_scores <= 0.5)).mean()),
                'news_count': len(recent_news),
                'sentiment_momentum': float(sentiment_scores.diff().mean()) if len(sentiment_scores) > 1 else 0.0
            }
            
            self.logger.info(f"Sentiment feature'ları hesaplandı: {len(recent_news)} haber")
            return features
            
        except Exception as e:
            self.logger.error(f"Sentiment feature hesaplama hatası: {e}")
            return self._get_empty_features()
    
    def _get_empty_features(self) -> Dict[str, float]:
        """Boş sentiment feature'ları döndür."""
        return {
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 1.0,
            'news_count': 0,
            'sentiment_momentum': 0.0
        }
    
    async def refresh_news_data(self) -> bool:
        """
        Haber verilerini yenile.
        
        Returns:
            bool: Başarılı olup olmadığı
        """
        try:
            self.logger.info("Haber verileri yenileniyor...")
            
            # Son güncelleme zamanını kontrol et
            if self._last_update and (datetime.now() - self._last_update).seconds < self.update_interval:
                self.logger.info("Güncelleme çok yakın zamanda yapılmış, atlanıyor")
                return True
            
            # Yeni haber verisi oluştur
            df = await self._create_news_csv()
            
            if not df.empty:
                self._last_update = datetime.now()
                self._news_cache = df
                self.logger.info("Haber verileri başarıyla yenilendi")
                return True
            else:
                self.logger.error("Haber verisi yenilenemedi")
                return False
                
        except Exception as e:
            self.logger.error(f"Haber verisi yenileme hatası: {e}")
            return False
    
    def get_news_summary(self) -> Dict[str, Any]:
        """Haber verisi özeti döndür."""
        try:
            if not self.csv_path.exists():
                return {"error": "CSV dosyası bulunamadı"}
            
            df = pd.read_csv(self.csv_path)
            
            summary = {
                "total_news": len(df),
                "latest_news": df['timestamp'].max() if 'timestamp' in df.columns else None,
                "sources": df['source'].value_counts().to_dict() if 'source' in df.columns else {},
                "sentiment_stats": {
                    "mean": float(df['sentiment'].mean()) if 'sentiment' in df.columns else 0.0,
                    "std": float(df['sentiment'].std()) if 'sentiment' in df.columns else 0.0,
                    "min": float(df['sentiment'].min()) if 'sentiment' in df.columns else 0.0,
                    "max": float(df['sentiment'].max()) if 'sentiment' in df.columns else 0.0
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Özet oluşturma hatası: {e}")
            return {"error": str(e)} 
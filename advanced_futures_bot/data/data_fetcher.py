import os
import time
import logging
from typing import Any, Dict, Optional
import pandas as pd
import requests
from binance.um_futures import UMFutures
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DataFetcher")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DataFetcher:
    """
    Production-level Binance Futures veri çekici.
    - Sağlam hata yönetimi, retry, rate limit
    - Veri ön işleme, logging
    - Whale API & On-chain: gerçek veya mock interface
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.symbol = config["binance"].get("symbol", "BTCUSDT") if config else "BTCUSDT"
        self.leverage = config["binance"].get("leverage", 10) if config else 10
        self.ohlcv_interval = config["data"].get("ohlcv_interval", "1m") if config else "1m"
        self.lookback = config["data"].get("lookback", 1000) if config else 1000
        self.order_book_depth = config["data"].get("order_book_depth", 50) if config else 50
        self.whale_api = config["data"].get("whale_api", True) if config else True
        self.onchain_metrics = config["data"].get("onchain_metrics", True) if config else True
        self.testnet = config["binance"].get("testnet", True) if config else True
        self.max_retries = config.get("max_retries", 5) if config else 5
        self.retry_delay = config.get("retry_delay", 2) if config else 2
        self.client = UMFutures(key=self.api_key, secret=self.api_secret, base_url=self._get_base_url())

    def _get_base_url(self):
        if self.testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    def _retry_request(self, func, *args, **kwargs):
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))
        logger.error(f"API call failed after {self.max_retries} attempts.")
        return None

    def fetch_ohlcv(self) -> pd.DataFrame:
        logger.info(f"Fetching OHLCV for {self.symbol} interval={self.ohlcv_interval} lookback={self.lookback}")
        klines = self._retry_request(self.client.klines, self.symbol, self.ohlcv_interval, limit=self.lookback)
        if not klines:
            logger.error("No OHLCV data fetched.")
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.fillna(method="ffill").fillna(0)
        logger.info(f"Fetched {len(df)} OHLCV rows.")
        return df

    def fetch_open_interest(self) -> float:
        logger.info(f"Fetching open interest for {self.symbol}")
        oi = self._retry_request(self.client.open_interest, self.symbol)
        if not oi or "openInterest" not in oi:
            logger.error("No open interest data.")
            return 0.0
        return float(oi["openInterest"])

    def fetch_funding_rate(self) -> float:
        logger.info(f"Fetching funding rate for {self.symbol}")
        funding = self._retry_request(self.client.funding_rate, symbol=self.symbol, limit=1)
        if not funding or not isinstance(funding, list) or not funding:
            logger.error("No funding rate data.")
            return 0.0
        return float(funding[0].get("fundingRate", 0.0))

    def fetch_order_book(self) -> Dict[str, Any]:
        logger.info(f"Fetching order book for {self.symbol} depth={self.order_book_depth}")
        ob = self._retry_request(self.client.depth, self.symbol, self.order_book_depth)
        if not ob or "bids" not in ob or "asks" not in ob:
            logger.error("No order book data.")
            return {"bids": [], "asks": []}
        return ob

    def fetch_whale_alert(self) -> Any:
        logger.info("Fetching whale alert (mock)")
        # Production: Gerçek API entegrasyonu burada yapılabilir
        # Şimdilik mock veri
        return {"whale_trades": []}

    def fetch_onchain_metrics(self) -> Any:
        logger.info("Fetching on-chain metrics (mock)")
        # Production: Gerçek API entegrasyonu burada yapılabilir
        # Şimdilik mock veri
        return {"onchain": {}}

    def fetch(self) -> Dict[str, Any]:
        """Tüm verileri tek seferde çeker."""
        logger.info("Fetching all data...")
        data = {
            "ohlcv": self.fetch_ohlcv(),
            "open_interest": self.fetch_open_interest(),
            "funding_rate": self.fetch_funding_rate(),
            "order_book": self.fetch_order_book(),
        }
        if self.whale_api:
            data["whale_alert"] = self.fetch_whale_alert()
        if self.onchain_metrics:
            data["onchain_metrics"] = self.fetch_onchain_metrics()
        logger.info("All data fetched.")
        return data 
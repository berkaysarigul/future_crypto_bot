import os
from typing import Any, Dict, Optional
import pandas as pd
import requests
from binance.um_futures import UMFutures
from dotenv import load_dotenv

load_dotenv()

class DataFetcher:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        self.symbol = config["binance"]["symbol"] if config else "BTCUSDT"
        self.leverage = config["binance"].get("leverage", 10) if config else 10
        self.ohlcv_interval = config["data"].get("ohlcv_interval", "1m") if config else "1m"
        self.lookback = config["data"].get("lookback", 1000) if config else 1000
        self.order_book_depth = config["data"].get("order_book_depth", 50) if config else 50
        self.whale_api = config["data"].get("whale_api", True) if config else True
        self.onchain_metrics = config["data"].get("onchain_metrics", True) if config else True
        self.client = UMFutures(key=self.api_key, secret=self.api_secret)

    def fetch_ohlcv(self) -> pd.DataFrame:
        klines = self.client.klines(self.symbol, self.ohlcv_interval, limit=self.lookback)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
        return df

    def fetch_open_interest(self) -> float:
        oi = self.client.open_interest(self.symbol)
        return float(oi["openInterest"])

    def fetch_funding_rate(self) -> float:
        funding = self.client.funding_rate(symbol=self.symbol, limit=1)
        return float(funding[0]["fundingRate"]) if funding else 0.0

    def fetch_order_book(self) -> Dict[str, Any]:
        ob = self.client.depth(self.symbol, limit=self.order_book_depth)
        return ob

    def fetch_whale_alert(self) -> Any:
        # Placeholder: Whale API entegrasyonu burada yapılacak
        return None

    def fetch_onchain_metrics(self) -> Any:
        # Placeholder: On-chain metrik API entegrasyonu burada yapılacak
        return None

    def fetch(self) -> Dict[str, Any]:
        """Tüm verileri tek seferde çeker."""
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
        return data 
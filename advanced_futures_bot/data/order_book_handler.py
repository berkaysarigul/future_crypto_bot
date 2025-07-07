from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

class OrderBookHandler:
    def __init__(self, order_book: Optional[Dict[str, Any]] = None):
        self.order_book = order_book

    def set_order_book(self, order_book: Dict[str, Any]):
        self.order_book = order_book

    def get_snapshot(self) -> Dict[str, Any]:
        """Ham order book snapshot'ı döner."""
        return self.order_book

    def depth_imbalance(self, depth: int = 10) -> float:
        """
        Belirli derinlikteki (örn. ilk 10 seviye) bid/ask hacim dengesizliğini hesaplar.
        Pozitif: Bid baskın, Negatif: Ask baskın.
        """
        if not self.order_book:
            return 0.0
        bids = self.order_book.get("bids", [])[:depth]
        asks = self.order_book.get("asks", [])[:depth]
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def spread(self) -> float:
        """En iyi bid ve ask arasındaki spread'i hesaplar."""
        if not self.order_book:
            return 0.0
        best_bid = float(self.order_book["bids"][0][0]) if self.order_book["bids"] else 0.0
        best_ask = float(self.order_book["asks"][0][0]) if self.order_book["asks"] else 0.0
        return best_ask - best_bid

    def liquidity_metrics(self, depth: int = 10) -> Dict[str, float]:
        """
        Belirli derinlikte toplam bid/ask hacmi ve likidite oranı döner.
        """
        if not self.order_book:
            return {"bid_liquidity": 0.0, "ask_liquidity": 0.0, "liq_ratio": 0.0}
        bids = self.order_book.get("bids", [])[:depth]
        asks = self.order_book.get("asks", [])[:depth]
        bid_liq = sum(float(b[1]) for b in bids)
        ask_liq = sum(float(a[1]) for a in asks)
        liq_ratio = bid_liq / ask_liq if ask_liq > 0 else 0.0
        return {"bid_liquidity": bid_liq, "ask_liquidity": ask_liq, "liq_ratio": liq_ratio}

    def process(self, depth: int = 10) -> Dict[str, Any]:
        """
        Order book'tan özet metrikleri çıkarır.
        """
        return {
            "snapshot": self.get_snapshot(),
            "depth_imbalance": self.depth_imbalance(depth),
            "spread": self.spread(),
            "liquidity": self.liquidity_metrics(depth)
        } 
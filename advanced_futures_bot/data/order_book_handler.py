import logging
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("OrderBookHandler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class OrderBookHandler:
    """
    Production-level order book snapshot & depth imbalance handler.
    - Veri validasyonu, outlier/edge-case yönetimi
    - Derinlik, spread, latency, logging
    """
    def __init__(self, order_book: Optional[Dict[str, Any]] = None):
        self.order_book = order_book or {"bids": [], "asks": []}
        self.last_latency = None

    def set_order_book(self, order_book: Dict[str, Any], latency: Optional[float] = None):
        self.order_book = order_book
        self.last_latency = latency
        logger.info(f"Order book updated. Latency: {latency if latency is not None else 'N/A'}s")

    def get_snapshot(self) -> Dict[str, Any]:
        """Ham order book snapshot'ı döner."""
        return self.order_book

    def _validate_depth(self, depth: int = 10) -> int:
        bids = self.order_book.get("bids", [])
        asks = self.order_book.get("asks", [])
        valid_depth = min(depth, len(bids), len(asks))
        if valid_depth < depth:
            logger.warning(f"Order book derinliği yetersiz: {valid_depth}/{depth}")
        return valid_depth

    def depth_imbalance(self, depth: int = 10) -> float:
        """
        Belirli derinlikteki bid/ask hacim dengesizliğini hesaplar.
        Pozitif: Bid baskın, Negatif: Ask baskın.
        """
        valid_depth = self._validate_depth(depth)
        if valid_depth == 0:
            logger.error("Order book derinliği yok.")
            return 0.0
        bids = self.order_book.get("bids", [])[:valid_depth]
        asks = self.order_book.get("asks", [])[:valid_depth]
        bid_vol = sum(float(b[1]) for b in bids)
        ask_vol = sum(float(a[1]) for a in asks)
        total = bid_vol + ask_vol
        if total == 0:
            logger.warning("Order book hacmi sıfır.")
            return 0.0
        return (bid_vol - ask_vol) / total

    def spread(self) -> float:
        """En iyi bid ve ask arasındaki spread'i hesaplar."""
        bids = self.order_book.get("bids", [])
        asks = self.order_book.get("asks", [])
        if not bids or not asks:
            logger.error("Order book spread hesaplanamıyor: bids/asks eksik.")
            return 0.0
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            if spread < 0:
                logger.warning(f"Negatif spread tespit edildi: {spread}")
                spread = 0.0
            return spread
        except Exception as e:
            logger.error(f"Spread hesaplama hatası: {e}")
            return 0.0

    def liquidity_metrics(self, depth: int = 10) -> Dict[str, float]:
        """
        Belirli derinlikte toplam bid/ask hacmi ve likidite oranı döner.
        """
        valid_depth = self._validate_depth(depth)
        if valid_depth == 0:
            return {"bid_liquidity": 0.0, "ask_liquidity": 0.0, "liq_ratio": 0.0}
        bids = self.order_book.get("bids", [])[:valid_depth]
        asks = self.order_book.get("asks", [])[:valid_depth]
        bid_liq = sum(float(b[1]) for b in bids)
        ask_liq = sum(float(a[1]) for a in asks)
        liq_ratio = bid_liq / ask_liq if ask_liq > 0 else 0.0
        return {"bid_liquidity": bid_liq, "ask_liquidity": ask_liq, "liq_ratio": liq_ratio}

    def outlier_check(self, threshold: float = 0.05) -> bool:
        """
        Outlier/edge-case kontrolü: Spread veya hacim anomali tespiti.
        """
        spread = self.spread()
        bids = self.order_book.get("bids", [])
        asks = self.order_book.get("asks", [])
        if not bids or not asks:
            return True
        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            if best_bid == 0:
                return True
            rel_spread = spread / best_bid
            if rel_spread > threshold:
                logger.warning(f"Outlier spread tespit edildi: {rel_spread:.4f}")
                return True
        except Exception as e:
            logger.error(f"Outlier kontrol hatası: {e}")
            return True
        return False

    def process(self, depth: int = 10) -> Dict[str, Any]:
        """
        Order book'tan özet metrikleri çıkarır.
        """
        return {
            "snapshot": self.get_snapshot(),
            "depth_imbalance": self.depth_imbalance(depth),
            "spread": self.spread(),
            "liquidity": self.liquidity_metrics(depth),
            "latency": self.last_latency,
            "outlier": self.outlier_check()
        } 
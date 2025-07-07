import logging
import time
from typing import Any, Dict, Optional, List, Tuple
import numpy as np

logger = logging.getLogger("ExecutionEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class ExecutionEngine:
    """
    Production-level realistic execution engine.
    - Order book depth simulation, realistic slippage, partial fill, TWAP
    - Latency, logging, exception handling
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.slippage_model = self.config.get("slippage_model", "linear")
        self.twap_enabled = self.config.get("twap_enabled", True)
        self.twap_duration = self.config.get("twap_duration", 60)
        self.twap_chunks = self.config.get("twap_chunks", 10)
        self.twap_threshold = self.config.get("twap_threshold", 1000)
        self.latency_ms = self.config.get("latency_ms", 10)
        self.min_fill_ratio = self.config.get("min_fill_ratio", 0.8)

    def calculate_slippage(self,
                          order_size: float,
                          order_book: Dict[str, List],
                          side: str = "buy") -> float:
        try:
            if side == "buy":
                levels = order_book.get("asks", [])
            else:
                levels = order_book.get("bids", [])
            if not levels:
                logger.warning("Order book boş, slippage 0 döndürülüyor")
                return 0.0
            remaining_size = order_size
            total_cost = 0.0
            weighted_price = 0.0
            for price, volume in levels:
                price = float(price)
                volume = float(volume)
                fill_size = min(remaining_size, volume)
                total_cost += fill_size * price
                weighted_price += fill_size
                remaining_size -= fill_size
                if remaining_size <= 0:
                    break
            if weighted_price > 0:
                avg_price = total_cost / weighted_price
                if side == "buy":
                    slippage = (avg_price - float(levels[0][0])) / float(levels[0][0])
                else:
                    slippage = (float(levels[0][0]) - avg_price) / float(levels[0][0])
                return slippage
            return 0.0
        except Exception as e:
            logger.error(f"Slippage hesaplama hatası: {e}")
            return 0.0

    def execute_twap(self,
                     order_size: float,
                     side: str,
                     order_book: Dict[str, List]) -> Dict[str, Any]:
        try:
            if not self.twap_enabled:
                return self.execute_market_order(order_size, side, order_book)
            chunk_size = order_size / self.twap_chunks
            total_filled = 0.0
            total_cost = 0.0
            fills = []
            for i in range(self.twap_chunks):
                result = self.execute_market_order(chunk_size, side, order_book)
                total_filled += result["filled_size"]
                total_cost += result["total_cost"]
                fills.append(result)
                if i < self.twap_chunks - 1:
                    time.sleep(self.twap_duration / self.twap_chunks)
            logger.info(f"TWAP execution tamamlandı: {total_filled}/{order_size} filled")
            return {
                "filled_size": total_filled,
                "total_cost": total_cost,
                "avg_price": total_cost / total_filled if total_filled > 0 else 0.0,
                "fills": fills,
                "execution_type": "twap"
            }
        except Exception as e:
            logger.error(f"TWAP execution hatası: {e}")
            return {"filled_size": 0.0, "total_cost": 0.0, "avg_price": 0.0, "fills": [], "execution_type": "error"}

    def execute_market_order(self,
                            order_size: float,
                            side: str,
                            order_book: Dict[str, List]) -> Dict[str, Any]:
        try:
            slippage = self.calculate_slippage(order_size, order_book, side)
            if side == "buy":
                base_price = float(order_book["asks"][0][0])
                execution_price = base_price * (1 + slippage)
            else:
                base_price = float(order_book["bids"][0][0])
                execution_price = base_price * (1 - slippage)
            fill_ratio = np.random.uniform(self.min_fill_ratio, 1.0)
            filled_size = order_size * fill_ratio
            logger.info(f"Market order: {side} {filled_size:.4f} @ {execution_price:.2f} (slippage={slippage:.4f})")
            return {
                "filled_size": filled_size,
                "total_cost": filled_size * execution_price,
                "avg_price": execution_price,
                "slippage": slippage,
                "fill_ratio": fill_ratio,
                "execution_type": "market"
            }
        except Exception as e:
            logger.error(f"Market order execution hatası: {e}")
            return {"filled_size": 0.0, "total_cost": 0.0, "avg_price": 0.0, "slippage": 0.0, "fill_ratio": 0.0, "execution_type": "error"}

    def run(self, order_size: float, side: str, order_book: Dict[str, List]) -> Dict[str, Any]:
        try:
            if self.twap_enabled and order_size > self.twap_threshold:
                return self.execute_twap(order_size, side, order_book)
            else:
                return self.execute_market_order(order_size, side, order_book)
        except Exception as e:
            logger.error(f"Execution engine hatası: {e}")
            return {"filled_size": 0.0, "total_cost": 0.0, "avg_price": 0.0, "execution_type": "error"} 
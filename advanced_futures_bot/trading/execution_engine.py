from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import time

class ExecutionEngine:
    """
    Realistic execution engine.
    - Order book depth simulation
    - Realistic slippage
    - Partial fill
    - TWAP execution
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.slippage_model = self.config.get("slippage_model", "linear")
        self.twap_enabled = self.config.get("twap_enabled", True)
        self.twap_duration = self.config.get("twap_duration", 60)  # seconds
        self.twap_chunks = self.config.get("twap_chunks", 10)

    def calculate_slippage(self,
                          order_size: float,
                          order_book: Dict[str, List],
                          side: str = "buy") -> float:
        """
        Realistic slippage hesaplama.
        """
        if side == "buy":
            levels = order_book.get("asks", [])
        else:
            levels = order_book.get("bids", [])
        
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

    def execute_twap(self,
                     order_size: float,
                     side: str,
                     order_book: Dict[str, List]) -> Dict[str, Any]:
        """
        TWAP execution.
        """
        if not self.twap_enabled:
            return self.execute_market_order(order_size, side, order_book)
        
        chunk_size = order_size / self.twap_chunks
        total_filled = 0.0
        total_cost = 0.0
        fills = []
        
        for i in range(self.twap_chunks):
            # Chunk execution
            result = self.execute_market_order(chunk_size, side, order_book)
            total_filled += result["filled_size"]
            total_cost += result["total_cost"]
            fills.append(result)
            
            # Wait between chunks
            if i < self.twap_chunks - 1:
                time.sleep(self.twap_duration / self.twap_chunks)
        
        return {
            "filled_size": total_filled,
            "total_cost": total_cost,
            "avg_price": total_cost / total_filled if total_filled > 0 else 0.0,
            "fills": fills,
            "execution_type": "twap"
        }

    def execute_market_order(self,
                            order_size: float,
                            side: str,
                            order_book: Dict[str, List]) -> Dict[str, Any]:
        """
        Market order execution with partial fill.
        """
        slippage = self.calculate_slippage(order_size, order_book, side)
        
        if side == "buy":
            base_price = float(order_book["asks"][0][0])
            execution_price = base_price * (1 + slippage)
        else:
            base_price = float(order_book["bids"][0][0])
            execution_price = base_price * (1 - slippage)
        
        # Partial fill simulation
        fill_ratio = np.random.uniform(0.8, 1.0)  # 80-100% fill
        filled_size = order_size * fill_ratio
        
        return {
            "filled_size": filled_size,
            "total_cost": filled_size * execution_price,
            "avg_price": execution_price,
            "slippage": slippage,
            "fill_ratio": fill_ratio,
            "execution_type": "market"
        }

    def run(self, order_size: float, side: str, order_book: Dict[str, List]) -> Dict[str, Any]:
        """
        Ana execution fonksiyonu.
        """
        if self.twap_enabled and order_size > self.config.get("twap_threshold", 1000):
            return self.execute_twap(order_size, side, order_book)
        else:
            return self.execute_market_order(order_size, side, order_book) 
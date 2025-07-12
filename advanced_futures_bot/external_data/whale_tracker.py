import logging
from typing import List, Dict
import aiohttp

class WhaleTracker:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = logging.getLogger("WhaleTracker")

    async def fetch_whale_alerts(self) -> List[Dict]:
        try:
            # TODO: Whale Alert API ile büyük transferleri çek
            return []
        except Exception as e:
            self.logger.error(f"Whale Alert çekme hatası: {e}")
            return [] 
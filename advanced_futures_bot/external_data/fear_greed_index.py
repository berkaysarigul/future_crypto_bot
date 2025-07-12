import logging
import aiohttp
from typing import Dict, Any

class FearGreedIndex:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.logger = logging.getLogger("FearGreedIndex")

    async def fetch_index(self) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data
            return {}
        except Exception as e:
            self.logger.error(f"Fear & Greed Index çekme hatası: {e}")
            return {} 
import logging
from typing import List, Dict
import aiohttp

class NewsScraper:
    def __init__(self, sources: List[str]):
        self.sources = sources
        self.logger = logging.getLogger("NewsScraper")

    async def fetch_news(self) -> List[Dict]:
        news = []
        try:
            async with aiohttp.ClientSession() as session:
                for url in self.sources:
                    try:
                        async with session.get(url) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                news.append(data)
                    except Exception as e:
                        self.logger.error(f"Kaynak {url} çekilemedi: {e}")
            self.logger.info("Haberler başarıyla çekildi.")
            return news
        except Exception as e:
            self.logger.error(f"Haber çekme hatası: {e}")
            return news 
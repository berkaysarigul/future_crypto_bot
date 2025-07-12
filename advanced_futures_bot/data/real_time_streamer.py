import asyncio
import websockets
import json
import os
from dotenv import load_dotenv
import logging

load_dotenv()

BINANCE_WS_URL = "wss://fstream.binance.com/ws/btcusdt@depth@100ms"

logger = logging.getLogger("real_time_streamer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RealTimeStreamer:
    def __init__(self):
        self.ws_url = BINANCE_WS_URL
        self.connection = None
        self.order_book = None

    async def connect(self):
        try:
            async with websockets.connect(self.ws_url) as websocket:
                self.connection = websocket
                logger.info("WebSocket bağlantısı kuruldu.")
                await self.listen()
        except Exception as e:
            logger.error(f"WebSocket bağlantı hatası: {e}")
            await asyncio.sleep(5)
            await self.connect()

    async def listen(self):
        try:
            async for message in self.connection:
                data = json.loads(message)
                await self.process_message(data)
        except Exception as e:
            logger.error(f"Dinleme sırasında hata: {e}")
            await self.connect()

    async def process_message(self, data):
        # L2 order book ve tick feed işleme
        self.order_book = data
        logger.debug(f"Order book güncellendi: {data}")

    async def start(self):
        await self.connect()

if __name__ == "__main__":
    streamer = RealTimeStreamer()
    asyncio.run(streamer.start()) 
import os
import logging
import aiohttp
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")

logger = logging.getLogger("alert_system")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class AlertSystem:
    def __init__(self):
        self.telegram_token = TELEGRAM_TOKEN
        self.telegram_chat_id = TELEGRAM_CHAT_ID
        self.discord_webhook = DISCORD_WEBHOOK

    async def send_telegram(self, message):
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram ayarları eksik.")
            return
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        data = {"chat_id": self.telegram_chat_id, "text": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=data) as resp:
                    if resp.status == 200:
                        logger.info("Telegram mesajı gönderildi.")
                    else:
                        logger.error(f"Telegram mesajı gönderilemedi: {resp.status}")
        except Exception as e:
            logger.error(f"Telegram hatası: {e}")

    async def send_discord(self, message):
        if not self.discord_webhook:
            logger.warning("Discord webhook eksik.")
            return
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.discord_webhook, json={"content": message}) as resp:
                    if resp.status == 204 or resp.status == 200:
                        logger.info("Discord mesajı gönderildi.")
                    else:
                        logger.error(f"Discord mesajı gönderilemedi: {resp.status}")
        except Exception as e:
            logger.error(f"Discord hatası: {e}")

    async def alert(self, message, level="info"):
        if level in ["error", "critical"]:
            await self.send_telegram(f"[ALERT] {message}")
            await self.send_discord(f"[ALERT] {message}")
        else:
            await self.send_telegram(message) 
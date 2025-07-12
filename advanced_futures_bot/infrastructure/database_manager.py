import os
import logging
import asyncpg
import aiosqlite
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("database_manager")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

DB_TYPE = os.getenv("DB_TYPE", "sqlite")
SQLITE_PATH = os.getenv("SQLITE_PATH", "trades.db")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "")

class DatabaseManager:
    def __init__(self):
        self.db_type = DB_TYPE
        self.conn = None

    async def connect(self):
        try:
            if self.db_type == "sqlite":
                self.conn = await aiosqlite.connect(SQLITE_PATH)
                logger.info("SQLite bağlantısı kuruldu.")
            elif self.db_type == "postgres":
                self.conn = await asyncpg.connect(POSTGRES_DSN)
                logger.info("PostgreSQL bağlantısı kuruldu.")
            else:
                raise ValueError("Desteklenmeyen DB tipi.")
        except Exception as e:
            logger.error(f"DB bağlantı hatası: {e}")
            raise

    async def save_trade(self, trade):
        try:
            if self.db_type == "sqlite":
                await self.conn.execute("INSERT INTO trades (data) VALUES (?)", (str(trade),))
                await self.conn.commit()
            elif self.db_type == "postgres":
                await self.conn.execute("INSERT INTO trades (data) VALUES ($1)", str(trade))
            logger.info("Trade kaydedildi.")
        except Exception as e:
            logger.error(f"Trade kaydedilemedi: {e}")

    async def close(self):
        if self.conn:
            await self.conn.close()
            logger.info("DB bağlantısı kapatıldı.") 
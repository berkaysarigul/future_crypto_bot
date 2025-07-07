import os
import time
import logging
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DeepSeekSentiment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DeepSeekSentiment:
    """
    Production-level DeepSeek LLM API entegrasyonu.
    - Sağlam hata yönetimi, retry, rate limit
    - Batch processing, logging
    - Config ve .env ile uyumlu
    """
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = (config or {}).get("api_url") or os.getenv("DEEPSEEK_API_URL") or "https://api.deepseek.com/v1/sentiment"
        self.max_retries = (config or {}).get("max_retries", 5)
        self.retry_delay = (config or {}).get("retry_delay", 2)
        self.batch_size = (config or {}).get("batch_size", 8)
        self.timeout = (config or {}).get("timeout", 10)
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _retry_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"DeepSeek API error {response.status_code}: {response.text}")
            except Exception as e:
                logger.warning(f"DeepSeek API call failed (attempt {attempt+1}/{self.max_retries}): {e}")
            time.sleep(self.retry_delay * (2 ** attempt))
        logger.error(f"DeepSeek API call failed after {self.max_retries} attempts.")
        return {"error": "API call failed"}

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch olarak metinlerde sentiment ve event tagging yapar.
        Args:
            texts: Analiz edilecek metin listesi
        Returns:
            Her metin için sentiment ve event tag içeren dict listesi
        """
        logger.info(f"Analyzing {len(texts)} texts with DeepSeek LLM API (batch_size={self.batch_size})")
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            payload = {
                "texts": batch,
                "features": ["sentiment", "event_tagging"]
            }
            batch_result = self._retry_request(payload)
            if "results" in batch_result:
                results.extend(batch_result["results"])
            else:
                # Hata durumunda her metin için error döndür
                for _ in batch:
                    results.append({"error": batch_result.get("error", "Unknown error")})
        logger.info(f"DeepSeek LLM API analysis complete. {len(results)} results.")
        return results 
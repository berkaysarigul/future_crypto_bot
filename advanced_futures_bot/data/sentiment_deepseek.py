import os
from typing import Any, Dict, List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

class DeepSeekSentiment:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.api_url = "https://api.deepseek.com/v1/sentiment"  # Örnek endpoint, gerçek endpoint ile değiştirilmeli

    def analyze(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        DeepSeek LLM API ile haber/sosyal medya metinlerinde sentiment ve event tagging yapar.
        Args:
            texts: Analiz edilecek metin listesi
        Returns:
            Her metin için sentiment ve event tag içeren dict listesi
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        results = []
        for text in texts:
            payload = {
                "text": text,
                "features": ["sentiment", "event_tagging"]
            }
            try:
                response = requests.post(self.api_url, json=payload, headers=headers, timeout=10)
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    results.append({"error": response.text})
            except Exception as e:
                results.append({"error": str(e)})
        return results 
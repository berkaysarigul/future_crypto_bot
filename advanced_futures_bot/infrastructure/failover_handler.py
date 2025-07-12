import logging
from typing import Any

class FailoverHandler:
    def __init__(self, alert_system: Any, health_checker: Any):
        self.alert_system = alert_system
        self.health_checker = health_checker
        self.logger = logging.getLogger("FailoverHandler")

    async def handle_failover(self, reason: str) -> None:
        try:
            self.logger.warning(f"Failover başlatıldı: {reason}")
            await self.alert_system.send_alert(f"Failover: {reason}")
            # TODO: Pozisyonları flat close et
        except Exception as e:
            self.logger.error(f"Failover hatası: {e}") 
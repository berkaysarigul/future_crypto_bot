import logging

logger = logging.getLogger("risk_manager")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RiskManager:
    def __init__(self, max_risk=0.01, stop_loss_pct=0.005):
        self.max_risk = max_risk  # Pozisyon başına risk (ör: 0.01 = %1)
        self.stop_loss_pct = stop_loss_pct  # Stop-loss yüzdesi

    async def calculate_position_size(self, balance, entry_price, stop_price):
        try:
            risk_amount = balance * self.max_risk
            position_size = risk_amount / abs(entry_price - stop_price)
            logger.info(f"Pozisyon büyüklüğü hesaplandı: {position_size}")
            return position_size
        except Exception as e:
            logger.error(f"Pozisyon büyüklüğü hesaplanamadı: {e}")
            return 0

    async def check_stop_loss(self, entry_price, current_price):
        try:
            if abs(current_price - entry_price) / entry_price >= self.stop_loss_pct:
                logger.info("Stop-loss tetiklendi.")
                return True
            return False
        except Exception as e:
            logger.error(f"Stop-loss kontrol hatası: {e}")
            return False

    async def check_risk_threshold(self, pnl, threshold):
        try:
            if pnl <= -threshold:
                logger.warning("Risk limiti aşıldı!")
                return True
            return False
        except Exception as e:
            logger.error(f"Risk limiti kontrol hatası: {e}")
            return False 
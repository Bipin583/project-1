"""
Sample Payment Processing Module
"""
class PaymentProcessor:
    def __init__(self, fee_rate: float = 0.02):
        self.fee_rate = fee_rate

    def calculate_fee(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError("Amount must be positive.")
        return round(amount * self.fee_rate, 2)

    def process_charge(self, card_number: str, amount: float) -> bool:
        if len(card_number) != 16 or not card_number.isdigit():
            return False
        return amount > 0

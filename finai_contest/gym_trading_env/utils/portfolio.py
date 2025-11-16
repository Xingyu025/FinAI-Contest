# finai_contest/gym_trading_env/utils/portfolio.py
from typing import Dict

class Portfolio:
    def __init__(self, position=0.0, value=1000.0, price=1.0):
        self.position = float(position)   # desired/target position
        self.value = float(value)         # portfolio monetary value
        self.price = float(price)         # last known price

    def get_portfolio_distribution(self) -> Dict:
        # return a simple distribution dict (placeholder)
        return {"cash": self.value, "position": self.position}

    def real_position(self, price: float) -> float:
        # naive: same as position
        return self.position

    def valorisation(self, price: float) -> float:
        # return portfolio value (no P&L sophistication in stub)
        self.price = price
        return float(self.value)

    def update_interest(self, borrow_interest_rate: float = 0.0):
        # naive: reduce value if negative position (placeholder)
        if self.position < 0:
            self.value -= abs(self.position) * borrow_interest_rate

class TargetPortfolio(Portfolio):
    def __init__(self, position=0.0, value=1000.0, price=1.0):
        super().__init__(position=position, value=value, price=price)

    def trade_to_position(self, position, price=None, trading_fees=0.0):
        # placeholder: set position and pretend fees are deducted from value
        if price is not None:
            self.price = price
        fee = abs(position - self.position) * trading_fees * (self.value)
        self.value -= fee
        self.position = position
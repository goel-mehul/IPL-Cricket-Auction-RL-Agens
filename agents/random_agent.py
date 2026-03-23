"""
random_agent.py — Uniform random BID/PASS. Sanity check baseline.
"""
import numpy as np


class RandomAgent:
    """Bids randomly with given probability."""

    def __init__(self, bid_prob: float = 0.3):
        self.bid_prob = bid_prob

    def act(self, obs: np.ndarray, budget: float, current_bid: float) -> int:
        if budget <= current_bid:
            return 0  # can't afford — pass
        return 1 if np.random.random() < self.bid_prob else 0

    def reset(self):
        pass
